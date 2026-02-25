const express = require("express");
const axios = require("axios");
const app = express();

// Middleware
app.use(express.json({ limit: "50mb" }));

// Configuration - Lấy từ environment variables
const CONFIG = {
    AZURE_ENDPOINT: process.env.AZURE_ENDPOINT,
    AZURE_API_KEY: process.env.AZURE_API_KEY,
    SERVICE_API_KEY: process.env.SERVICE_API_KEY,
    PORT: process.env.PORT || 8080,
    ANTHROPIC_VERSION: "2023-06-01",
    AZURE_DEPLOYMENT_NAME: process.env.AZURE_DEPLOYMENT_NAME || "claude-opus-4-5",
};

const MODEL_NAMES_TO_MAP = ["gpt-4", "gpt-4.1", "gpt-4o", "claude-opus-4-5", "claude-4.5-opus-high", "claude-4-opus", "claude-3-opus", "claude-3-sonnet", "claude-3-haiku"];

function mapModelToDeployment(modelName) {
    if (!modelName) {
        return CONFIG.AZURE_DEPLOYMENT_NAME;
    }
    if (MODEL_NAMES_TO_MAP.includes(modelName)) {
        return CONFIG.AZURE_DEPLOYMENT_NAME;
    }
    if (process.env.AZURE_DEPLOYMENT_NAME) {
        return CONFIG.AZURE_DEPLOYMENT_NAME;
    }
    return modelName;
}

function fixImageTurns(messages) {
    if (!Array.isArray(messages)) return messages;
    const fixed = [];
    for (let i = 0; i < messages.length; i++) {
        const msg = messages[i];
        const hasImage =
            msg.content &&
            Array.isArray(msg.content) &&
            msg.content.some(c => c.type === "image_url");
        if (hasImage && i > 0 && fixed[fixed.length - 1].role === "assistant") {
            fixed.pop();
        }
        fixed.push(msg);
    }
    return fixed;
}

// CORS middleware
app.use((req, res, next) => {
    res.header("Access-Control-Allow-Origin", "*");
    res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.header("Access-Control-Allow-Headers", "Content-Type, Authorization, x-api-key, anthropic-version");
    if (req.method === "OPTIONS") {
        return res.sendStatus(200);
    }
    next();
});

app.use((req, res, next) => {
    console.log(`[${req.method}] ${req.path}`);
    next();
});

function requireAuth(req, res, next) {
    if (req.method === "OPTIONS" || req.path === "/health" || req.path === "/") {
        return next();
    }
    if (!CONFIG.SERVICE_API_KEY) {
        console.error("[ERROR] SERVICE_API_KEY not configured");
        return res.status(500).json({
            error: {
                message: "SERVICE_API_KEY not configured",
                type: "configuration_error",
            },
        });
    }
    const authHeader = req.headers.authorization;
    if (!authHeader) {
        console.error("[ERROR] Missing Authorization header");
        return res.status(401).json({
            error: {
                message: "Missing Authorization header",
                type: "authentication_error",
            },
        });
    }
    let token = authHeader;
    if (authHeader.startsWith("Bearer ")) {
        token = authHeader.substring(7);
    }
    if (token !== CONFIG.SERVICE_API_KEY) {
        console.error("[ERROR] Invalid API key provided");
        return res.status(401).json({
            error: {
                message: "Invalid API key",
                type: "authentication_error",
            },
        });
    }
    next();
}

function transformRequest(openAIRequest) {
    const { messages, model, max_tokens, temperature, stream, role, content, input, user, tools, tool_choice, ...rest } = openAIRequest;

    let anthropicMessages;

    if (messages && Array.isArray(messages)) {
        anthropicMessages = messages
            .filter((msg) => msg && (msg.content || msg.content === ""))
            .map((msg) => {
                if (msg.role === "system") {
                    return {
                        role: "user",
                        content: typeof msg.content === "string" ? `System: ${msg.content}` : msg.content,
                    };
                }
                
                // Handle tool/function role messages (tool results)
                if (msg.role === "tool" || msg.role === "function") {
                    return {
                        role: "user",
                        content: [{
                            type: "tool_result",
                            tool_use_id: msg.tool_call_id || msg.name,
                            content: typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content)
                        }]
                    };
                }
                
                // Handle assistant messages with tool_calls
                if (msg.role === "assistant" && msg.tool_calls && Array.isArray(msg.tool_calls)) {
                    const content = [];
                    if (msg.content) {
                        content.push({ type: "text", text: msg.content });
                    }
                    for (const toolCall of msg.tool_calls) {
                        content.push({
                            type: "tool_use",
                            id: toolCall.id,
                            name: toolCall.function?.name || toolCall.name,
                            input: typeof toolCall.function?.arguments === "string" 
                                ? JSON.parse(toolCall.function.arguments) 
                                : toolCall.function?.arguments || {}
                        });
                    }
                    return {
                        role: "assistant",
                        content: content
                    };
                }
                
                const role = msg.role === "assistant" ? "assistant" : "user";
                return {
                    role: role,
                    content: msg.content,
                };
            });
    } else if (role && content) {
        anthropicMessages = [
            {
                role: role === "system" ? "user" : role,
                content: role === "system" ? `System: ${content}` : content,
            },
        ];
    } else if (input) {
        if (Array.isArray(input)) {
            anthropicMessages = input
                .filter((msg) => msg && (msg.content !== undefined || msg.content === ""))
                .map((msg) => {
                    if (msg.role === "system") {
                        return {
                            role: "user",
                            content: typeof msg.content === "string" ? `System: ${msg.content}` : msg.content,
                        };
                    }
                    const role = msg.role === "assistant" ? "assistant" : "user";
                    return {
                        role: role,
                        content: msg.content !== undefined ? msg.content : String(msg),
                    };
                });
        } else {
            anthropicMessages = [
                {
                    role: user || "user",
                    content: input,
                },
            ];
        }
    } else if (content) {
        anthropicMessages = [
            {
                role: "user",
                content: content,
            },
        ];
    } else {
        throw new Error("Invalid request format: missing messages, role/content, input, or content field");
    }

    if (!anthropicMessages || anthropicMessages.length === 0) {
        throw new Error("Invalid request: no valid messages found");
    }

    const azureModelName = mapModelToDeployment(model);
    const anthropicRequest = {
        model: azureModelName,
        messages: anthropicMessages,
        max_tokens: max_tokens || 8192,
    };

    if (temperature !== undefined) {
        anthropicRequest.temperature = temperature;
    }
    if (stream !== undefined) {
        anthropicRequest.stream = stream;
    }

    // Transform OpenAI tools to Anthropic tools format
    if (tools && Array.isArray(tools) && tools.length > 0) {
        anthropicRequest.tools = tools.map(tool => {
            if (tool.type === "function") {
                return {
                    name: tool.function.name,
                    description: tool.function.description || "",
                    input_schema: tool.function.parameters || { type: "object", properties: {} }
                };
            }
            return tool;
        });
    }

    // Handle tool_choice
    if (tool_choice) {
        if (tool_choice === "auto") {
            anthropicRequest.tool_choice = { type: "auto" };
        } else if (tool_choice === "none") {
            // Don't send tools if none is specified
            delete anthropicRequest.tools;
        } else if (tool_choice === "required") {
            anthropicRequest.tool_choice = { type: "any" };
        } else if (typeof tool_choice === "object" && tool_choice.function?.name) {
            anthropicRequest.tool_choice = { type: "tool", name: tool_choice.function.name };
        }
    }

    const supportedFields = ["metadata", "stop_sequences", "top_p", "top_k"];
    for (const field of supportedFields) {
        if (rest[field] !== undefined) {
            anthropicRequest[field] = rest[field];
        }
    }

    if (rest.system !== undefined) {
        if (Array.isArray(rest.system)) {
            anthropicRequest.system = rest.system;
        } else if (typeof rest.system === "string") {
            anthropicRequest.system = rest.system;
        } else {
            anthropicRequest.system = String(rest.system);
        }
    }

    return anthropicRequest;
}

function transformResponse(anthropicResponse) {
    const { content, id, model, stop_reason, usage } = anthropicResponse;

    const response = {
        id: id,
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: [
            {
                index: 0,
                message: {
                    role: "assistant",
                    content: null,
                },
                finish_reason: stop_reason === "tool_use" ? "tool_calls" : stop_reason,
            },
        ],
        usage: {
            prompt_tokens: usage?.input_tokens || 0,
            completion_tokens: usage?.output_tokens || 0,
            total_tokens: (usage?.input_tokens || 0) + (usage?.output_tokens || 0),
        },
    };

    // Process content blocks
    let textContent = "";
    const toolCalls = [];

    for (const block of content) {
        if (block.type === "text") {
            textContent += block.text;
        } else if (block.type === "tool_use") {
            toolCalls.push({
                id: block.id,
                type: "function",
                function: {
                    name: block.name,
                    arguments: JSON.stringify(block.input)
                }
            });
        }
    }

    if (textContent) {
        response.choices[0].message.content = textContent;
    }

    if (toolCalls.length > 0) {
        response.choices[0].message.tool_calls = toolCalls;
    }

    return response;
}

app.get("/", (req, res) => {
    res.json({
        status: "running",
        name: "Azure Anthropic Proxy for Cursor",
        version: "2.0.0",
        endpoints: {
            health: "/health",
            chat_cursor: "/chat/completions",
            chat_openai: "/v1/chat/completions",
            chat_anthropic: "/v1/messages",
        },
    });
});

app.get("/health", (req, res) => {
    res.json({
        status: "ok",
        timestamp: new Date().toISOString(),
        apiKeyConfigured: !!CONFIG.AZURE_API_KEY,
        port: CONFIG.PORT,
    });
});

app.post("/chat/completions", requireAuth, async (req, res) => {
    console.log("[REQUEST /chat/completions]", new Date().toISOString());
    console.log("Model:", req.body?.model, "Stream:", req.body?.stream, "Tools:", req.body?.tools?.length || 0);

    try {
        if (!CONFIG.AZURE_API_KEY) {
            return res.status(500).json({ error: { message: "Azure API key not configured", type: "configuration_error" } });
        }
        if (!CONFIG.AZURE_ENDPOINT) {
            return res.status(500).json({ error: { message: "Azure endpoint not configured", type: "configuration_error" } });
        }
        if (!req.body) {
            return res.status(400).json({ error: { message: "Invalid request: empty body", type: "invalid_request_error" } });
        }

        const hasMessages = req.body.messages && Array.isArray(req.body.messages);
        const hasRoleContent = req.body.role && req.body.content;
        const hasInput = req.body.input && (Array.isArray(req.body.input) || typeof req.body.input === "string");
        const hasContent = req.body.content;

        if (!hasMessages && !hasRoleContent && !hasInput && !hasContent) {
            return res.status(400).json({ error: { message: "Invalid request: must include messages", type: "invalid_request_error" } });
        }

        const isStreaming = req.body.stream === true;

        let anthropicRequest;
        try {
            if (req.body.messages) {
                req.body.messages = fixImageTurns(req.body.messages);
            }
            anthropicRequest = transformRequest(req.body);
            console.log("[AZURE] Using model/deployment:", anthropicRequest.model);
            console.log("[AZURE] Tools count:", anthropicRequest.tools?.length || 0);
        } catch (transformError) {
            console.error("[ERROR] Failed to transform request:", transformError);
            return res.status(400).json({ error: { message: "Failed to transform request: " + transformError.message, type: "transform_error" } });
        }

        const response = await axios.post(CONFIG.AZURE_ENDPOINT, anthropicRequest, {
            headers: {
                "Content-Type": "application/json",
                "x-api-key": CONFIG.AZURE_API_KEY,
                "anthropic-version": CONFIG.ANTHROPIC_VERSION,
            },
            timeout: 300000,
            responseType: isStreaming ? "stream" : "json",
            validateStatus: (status) => status < 600,
        });

        console.log("[AZURE] Response status:", response.status);

        if (response.status >= 400) {
            let errorMessage = "Azure API error";
            if (response.data) {
                if (isStreaming && typeof response.data.pipe === "function") {
                    let errorBuffer = "";
                    await new Promise((resolve) => {
                        response.data.on("data", (chunk) => { errorBuffer += chunk.toString(); });
                        response.data.on("end", resolve);
                        response.data.on("error", resolve);
                    });
                    try {
                        const parsed = JSON.parse(errorBuffer);
                        errorMessage = parsed?.error?.message || errorBuffer;
                    } catch (e) {
                        errorMessage = errorBuffer;
                    }
                } else if (response.data.error) {
                    errorMessage = response.data.error.message;
                }
            }
            return res.status(response.status).json({ error: { message: errorMessage, type: "
