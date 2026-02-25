const express = require("express");
const axios = require("axios");
const app = express();

app.use(express.json({ limit: "50mb" }));

const CONFIG = {
    // Claude (Anthropic) on Azure
    AZURE_ENDPOINT: process.env.AZURE_ENDPOINT,
    AZURE_API_KEY: process.env.AZURE_API_KEY,
    AZURE_DEPLOYMENT_NAME: process.env.AZURE_DEPLOYMENT_NAME || "claude-opus-4-5",
    ANTHROPIC_VERSION: "2023-06-01",
    
    // GPT (OpenAI) on Azure - Responses API
    AZURE_OPENAI_ENDPOINT: process.env.AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY: process.env.AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_MODEL: process.env.AZURE_OPENAI_MODEL || "gpt-5.3-codex",
    
    // Service auth
    SERVICE_API_KEY: process.env.SERVICE_API_KEY,
    PORT: process.env.PORT || 8080,
};

// Model patterns for routing
const CLAUDE_MODELS = ["claude", "anthropic"];
const GPT_MODELS = ["gpt", "openai", "codex", "o1", "o3"];

function isGPTModel(modelName) {
    if (!modelName) return false;
    const lower = modelName.toLowerCase();
    return GPT_MODELS.some(pattern => lower.includes(pattern));
}

function isClaudeModel(modelName) {
    if (!modelName) return true; // Default to Claude
    const lower = modelName.toLowerCase();
    return CLAUDE_MODELS.some(pattern => lower.includes(pattern));
}

function fixImageTurns(messages) {
    if (!Array.isArray(messages)) return messages;
    const fixed = [];
    for (let i = 0; i < messages.length; i++) {
        const msg = messages[i];
        const hasImage = msg.content && Array.isArray(msg.content) && msg.content.some(c => c.type === "image_url");
        if (hasImage && i > 0 && fixed[fixed.length - 1]?.role === "assistant") {
            fixed.pop();
        }
        fixed.push(msg);
    }
    return fixed;
}

app.use((req, res, next) => {
    res.header("Access-Control-Allow-Origin", "*");
    res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.header("Access-Control-Allow-Headers", "Content-Type, Authorization, x-api-key, anthropic-version");
    if (req.method === "OPTIONS") return res.sendStatus(200);
    next();
});

app.use((req, res, next) => {
    console.log(`[${req.method}] ${req.path}`);
    next();
});

function requireAuth(req, res, next) {
    if (req.method === "OPTIONS" || req.path === "/health" || req.path === "/") return next();
    if (!CONFIG.SERVICE_API_KEY) {
        return res.status(500).json({ error: { message: "SERVICE_API_KEY not configured", type: "configuration_error" } });
    }
    const authHeader = req.headers.authorization;
    if (!authHeader) {
        return res.status(401).json({ error: { message: "Missing Authorization header", type: "authentication_error" } });
    }
    let token = authHeader.startsWith("Bearer ") ? authHeader.substring(7) : authHeader;
    if (token !== CONFIG.SERVICE_API_KEY) {
        return res.status(401).json({ error: { message: "Invalid API key", type: "authentication_error" } });
    }
    next();
}

// ============================================================================
// CLAUDE (Anthropic) TRANSFORMATIONS
// ============================================================================

function transformRequestForClaude(openAIRequest) {
    const { messages, model, max_tokens, temperature, stream, role, content, input, user, tools, tool_choice, ...rest } = openAIRequest;

    let anthropicMessages;

    if (messages && Array.isArray(messages)) {
        anthropicMessages = messages
            .filter((msg) => msg && (msg.content || msg.content === "" || msg.tool_calls))
            .map((msg) => {
                if (msg.role === "system") {
                    return { role: "user", content: typeof msg.content === "string" ? `System: ${msg.content}` : msg.content };
                }
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
                if (msg.role === "assistant" && msg.tool_calls && Array.isArray(msg.tool_calls)) {
                    const contentBlocks = [];
                    if (msg.content) {
                        contentBlocks.push({ type: "text", text: msg.content });
                    }
                    for (const toolCall of msg.tool_calls) {
                        let args = {};
                        try {
                            args = typeof toolCall.function?.arguments === "string"
                                ? JSON.parse(toolCall.function.arguments)
                                : toolCall.function?.arguments || {};
                        } catch (e) {
                            args = {};
                        }
                        contentBlocks.push({
                            type: "tool_use",
                            id: toolCall.id,
                            name: toolCall.function?.name || toolCall.name,
                            input: args
                        });
                    }
                    return { role: "assistant", content: contentBlocks };
                }
                return { role: msg.role === "assistant" ? "assistant" : "user", content: msg.content };
            });
    } else if (role && content) {
        anthropicMessages = [{ role: role === "system" ? "user" : role, content: role === "system" ? `System: ${content}` : content }];
    } else if (input) {
        if (Array.isArray(input)) {
            anthropicMessages = input.filter((msg) => msg && msg.content !== undefined).map((msg) => {
                if (msg.role === "system") return { role: "user", content: `System: ${msg.content}` };
                return { role: msg.role === "assistant" ? "assistant" : "user", content: msg.content };
            });
        } else {
            anthropicMessages = [{ role: user || "user", content: input }];
        }
    } else if (content) {
        anthropicMessages = [{ role: "user", content: content }];
    } else {
        throw new Error("Invalid request format");
    }

    if (!anthropicMessages || anthropicMessages.length === 0) {
        throw new Error("Invalid request: no valid messages found");
    }

    const anthropicRequest = {
        model: CONFIG.AZURE_DEPLOYMENT_NAME,
        messages: anthropicMessages,
        max_tokens: max_tokens || 8192,
    };

    if (temperature !== undefined) anthropicRequest.temperature = temperature;
    if (stream !== undefined) anthropicRequest.stream = stream;

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

    if (tool_choice) {
        if (tool_choice === "auto") {
            anthropicRequest.tool_choice = { type: "auto" };
        } else if (tool_choice === "none") {
            delete anthropicRequest.tools;
        } else if (tool_choice === "required") {
            anthropicRequest.tool_choice = { type: "any" };
        } else if (typeof tool_choice === "object" && tool_choice.function?.name) {
            anthropicRequest.tool_choice = { type: "tool", name: tool_choice.function.name };
        }
    }

    const supportedFields = ["metadata", "stop_sequences", "top_p", "top_k"];
    for (const field of supportedFields) {
        if (rest[field] !== undefined) anthropicRequest[field] = rest[field];
    }

    if (rest.system !== undefined) {
        anthropicRequest.system = Array.isArray(rest.system) ? rest.system : String(rest.system);
    }

    return anthropicRequest;
}

function transformClaudeResponse(anthropicResponse) {
    const { content, id, model, stop_reason, usage } = anthropicResponse;

    const response = {
        id: id,
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: [{
            index: 0,
            message: { role: "assistant", content: null },
            finish_reason: stop_reason === "tool_use" ? "tool_calls" : (stop_reason === "end_turn" ? "stop" : stop_reason),
        }],
        usage: {
            prompt_tokens: usage?.input_tokens || 0,
            completion_tokens: usage?.output_tokens || 0,
            total_tokens: (usage?.input_tokens || 0) + (usage?.output_tokens || 0),
        },
    };

    let textContent = "";
    const toolCalls = [];

    if (content && Array.isArray(content)) {
        for (const block of content) {
            if (block.type === "text") {
                textContent += block.text;
            } else if (block.type === "tool_use") {
                toolCalls.push({
                    id: block.id,
                    type: "function",
                    function: { name: block.name, arguments: JSON.stringify(block.input) }
                });
            }
        }
    }

    if (textContent) response.choices[0].message.content = textContent;
    if (toolCalls.length > 0) response.choices[0].message.tool_calls = toolCalls;

    return response;
}

// ============================================================================
// GPT (Azure OpenAI Responses API) TRANSFORMATIONS
// ============================================================================

function transformRequestForGPT(openAIRequest) {
    const { messages, input, model, max_tokens, temperature, stream, tools, tool_choice, ...rest } = openAIRequest;

    // Convert messages array to Responses API format
    // The Responses API uses "input" which can be a string or array of content items
    let inputItems = [];
    let systemPrompt = null;

    if (messages && Array.isArray(messages)) {
        for (const msg of messages) {
            if (!msg) continue;
            
            if (msg.role === "system") {
                // Collect system messages
                const text = typeof msg.content === "string" ? msg.content : 
                    (Array.isArray(msg.content) ? msg.content.map(c => c.text || c.content || "").join("\n") : "");
                systemPrompt = systemPrompt ? systemPrompt + "\n" + text : text;
            } else if (msg.role === "user") {
                // User message
                let content = msg.content;
                if (typeof content === "string") {
                    inputItems.push({ type: "message", role: "user", content: content });
                } else if (Array.isArray(content)) {
                    // Handle content array (text, images, etc.)
                    const textParts = content.filter(c => c.type === "text").map(c => c.text).join("\n");
                    if (textParts) {
                        inputItems.push({ type: "message", role: "user", content: textParts });
                    }
                }
            } else if (msg.role === "assistant") {
                // Assistant message
                if (msg.tool_calls && Array.isArray(msg.tool_calls)) {
                    // Handle tool calls
                    for (const tc of msg.tool_calls) {
                        inputItems.push({
                            type: "function_call",
                            call_id: tc.id,
                            name: tc.function?.name,
                            arguments: tc.function?.arguments || "{}"
                        });
                    }
                }
                if (msg.content) {
                    inputItems.push({ type: "message", role: "assistant", content: msg.content });
                }
            } else if (msg.role === "tool") {
                // Tool result
                inputItems.push({
                    type: "function_call_output",
                    call_id: msg.tool_call_id,
                    output: typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content)
                });
            }
        }
    }

    // Some clients send an `input` field instead of `messages`.
    if ((!inputItems.length || !messages) && input !== undefined) {
        if (typeof input === "string") {
            inputItems.push({ type: "message", role: "user", content: input });
        } else if (Array.isArray(input)) {
            for (const item of input) {
                if (!item) continue;
                if (typeof item === "string") {
                    inputItems.push({ type: "message", role: "user", content: item });
                    continue;
                }

                if (item.type === "message" && item.content) {
                    inputItems.push({
                        type: "message",
                        role: item.role || "user",
                        content: item.content,
                    });
                    continue;
                }

                if (item.role && item.content) {
                    inputItems.push({
                        type: "message",
                        role: item.role,
                        content: item.content,
                    });
                }
            }
        }
    }

    if (inputItems.length === 0) {
        throw new Error("No usable input/messages for GPT request");
    }

    // Build the Responses API request
    const gptRequest = {
        model: CONFIG.AZURE_OPENAI_MODEL,
        input: inputItems,
        max_output_tokens: max_tokens || 8192,
    };

    // Add system instructions if present
    if (systemPrompt) {
        gptRequest.instructions = systemPrompt;
    }

    if (temperature !== undefined) gptRequest.temperature = temperature;
    if (stream !== undefined) gptRequest.stream = stream;

    // Transform tools to Responses API format
    if (tools && Array.isArray(tools) && tools.length > 0) {
        gptRequest.tools = tools
            .filter(tool => tool && (tool.function?.name || tool.name))
            .map(tool => {
                if (tool.type === "function" && tool.function) {
                    return {
                        type: "function",
                        name: tool.function.name,
                        description: tool.function.description || "",
                        parameters: tool.function.parameters || { type: "object", properties: {} }
                    };
                } else if (tool.name) {
                    return {
                        type: "function",
                        name: tool.name,
                        description: tool.description || "",
                        parameters: tool.parameters || tool.input_schema || { type: "object", properties: {} }
                    };
                }
                return null;
            })
            .filter(Boolean);
    }

    // Handle tool_choice
    if (tool_choice) {
        if (tool_choice === "auto") {
            gptRequest.tool_choice = "auto";
        } else if (tool_choice === "none") {
            gptRequest.tool_choice = "none";
        } else if (tool_choice === "required") {
            gptRequest.tool_choice = "required";
        } else if (typeof tool_choice === "object" && tool_choice.function?.name) {
            gptRequest.tool_choice = { type: "function", function: { name: tool_choice.function.name } };
        }
    }

    return gptRequest;
}

function transformGPTResponse(gptResponse) {
    // Responses API returns a different structure
    const { id, model, output, usage, status } = gptResponse;

    const response = {
        id: id || "chatcmpl-" + Date.now(),
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: model || CONFIG.AZURE_OPENAI_MODEL,
        choices: [{
            index: 0,
            message: { role: "assistant", content: null },
            finish_reason: "stop",
        }],
        usage: {
            prompt_tokens: usage?.input_tokens || 0,
            completion_tokens: usage?.output_tokens || 0,
            total_tokens: (usage?.input_tokens || 0) + (usage?.output_tokens || 0),
        },
    };

    let textContent = "";
    const toolCalls = [];

    // Parse output array from Responses API
    if (output && Array.isArray(output)) {
        for (const item of output) {
            if (item.type === "message" && item.content) {
                // Handle message content
                if (Array.isArray(item.content)) {
                    for (const c of item.content) {
                        if (c.type === "output_text" || c.type === "text") {
                            textContent += c.text || "";
                        }
                    }
                } else if (typeof item.content === "string") {
                    textContent += item.content;
                }
            } else if (item.type === "function_call") {
                toolCalls.push({
                    id: item.call_id || item.id || "call_" + Date.now(),
                    type: "function",
                    function: {
                        name: item.name,
                        arguments: item.arguments || "{}"
                    }
                });
            }
        }
    }

    if (textContent) response.choices[0].message.content = textContent;
    if (toolCalls.length > 0) {
        response.choices[0].message.tool_calls = toolCalls;
        response.choices[0].finish_reason = "tool_calls";
    }

    return response;
}

// ============================================================================
// ENDPOINTS
// ============================================================================

app.get("/", (req, res) => {
    res.json({
        status: "running",
        name: "Azure Multi-Model Proxy (Claude + GPT)",
        version: "3.0.0",
        endpoints: { 
            health: "/health", 
            chat_cursor: "/chat/completions", 
            chat_openai: "/v1/chat/completions", 
            chat_anthropic: "/v1/messages" 
        },
        models: {
            claude: CONFIG.AZURE_ENDPOINT ? "configured" : "not configured",
            gpt: CONFIG.AZURE_OPENAI_ENDPOINT ? "configured" : "not configured"
        }
    });
});

app.get("/health", (req, res) => {
    res.json({ 
        status: "ok", 
        timestamp: new Date().toISOString(), 
        claude: !!CONFIG.AZURE_API_KEY,
        gpt: !!CONFIG.AZURE_OPENAI_API_KEY,
        port: CONFIG.PORT 
    });
});

app.post("/chat/completions", requireAuth, async (req, res) => {
    const modelName = req.body?.model || "";
    const useGPT = isGPTModel(modelName);
    
    console.log("[REQUEST /chat/completions]", new Date().toISOString());
    console.log("Model:", modelName, "Route:", useGPT ? "GPT" : "Claude", "Stream:", req.body?.stream, "Tools:", req.body?.tools?.length || 0);

    try {
        if (!req.body) {
            return res.status(400).json({ error: { message: "Invalid request: empty body", type: "invalid_request_error" } });
        }

        const hasMessages = req.body.messages && Array.isArray(req.body.messages);
        if (!hasMessages && !req.body.content && !req.body.input) {
            return res.status(400).json({ error: { message: "Invalid request: must include messages", type: "invalid_request_error" } });
        }

        if (req.body.messages) req.body.messages = fixImageTurns(req.body.messages);

        if (useGPT) {
            await handleGPTRequest(req, res);
        } else {
            await handleClaudeRequest(req, res);
        }
    } catch (error) {
        console.error("[ERROR]", error.message);
        if (error.response) {
            return res.status(error.response.status).json({ error: { message: error.response.data?.error?.message || error.message, type: "api_error" } });
        } else if (error.request) {
            return res.status(503).json({ error: { message: "Unable to reach API", type: "connection_error" } });
        } else {
            return res.status(500).json({ error: { message: error.message, type: "proxy_error" } });
        }
    }
});

async function handleClaudeRequest(req, res) {
    if (!CONFIG.AZURE_API_KEY) {
        return res.status(500).json({ error: { message: "Claude API key not configured", type: "configuration_error" } });
    }
    if (!CONFIG.AZURE_ENDPOINT) {
        return res.status(500).json({ error: { message: "Claude endpoint not configured", type: "configuration_error" } });
    }

    const isStreaming = req.body.stream === true;

    let anthropicRequest;
    try {
        anthropicRequest = transformRequestForClaude(req.body);
        console.log("[CLAUDE] Model:", anthropicRequest.model, "Tools:", anthropicRequest.tools?.length || 0);
    } catch (transformError) {
        console.error("[ERROR] Transform failed:", transformError.message);
        return res.status(400).json({ error: { message: "Transform error: " + transformError.message, type: "transform_error" } });
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

    console.log("[CLAUDE] Response status:", response.status);

    if (response.status >= 400) {
        let errorMessage = "Claude API error";
        if (response.data) {
            if (isStreaming && typeof response.data.pipe === "function") {
                let errorBuffer = "";
                await new Promise((resolve) => {
                    response.data.on("data", (chunk) => { errorBuffer += chunk.toString(); });
                    response.data.on("end", resolve);
                    response.data.on("error", resolve);
                });
                try { errorMessage = JSON.parse(errorBuffer)?.error?.message || errorBuffer; } catch (e) { errorMessage = errorBuffer; }
            } else if (response.data.error) {
                errorMessage = response.data.error.message;
            }
        }
        return res.status(response.status).json({ error: { message: errorMessage, type: "api_error" } });
    }

    if (isStreaming) {
        handleClaudeStreaming(req, res, response);
    } else {
        try {
            const openAIResponse = transformClaudeResponse(response.data);
            res.json(openAIResponse);
        } catch (transformError) {
            console.error("[ERROR] Response transform failed:", transformError.message);
            return res.status(500).json({ error: { message: "Transform error", type: "transform_error" } });
        }
    }
}

function handleClaudeStreaming(req, res, response) {
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");

    let buffer = "";
    const contentBlocks = {};
    let currentToolCallIndex = 0;

    response.data.on("data", (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const data = line.slice(6).trim();
            if (data === "[DONE]") {
                res.write("data: [DONE]\n\n");
                continue;
            }

            try {
                const event = JSON.parse(data);

                if (event.type === "content_block_start") {
                    const idx = event.index;
                    contentBlocks[idx] = event.content_block;

                    if (event.content_block.type === "tool_use") {
                        const toolChunk = {
                            id: "chatcmpl-" + Date.now(),
                            object: "chat.completion.chunk",
                            created: Math.floor(Date.now() / 1000),
                            model: req.body.model || "claude-opus-4-5",
                            choices: [{
                                index: 0,
                                delta: {
                                    tool_calls: [{
                                        index: currentToolCallIndex,
                                        id: event.content_block.id,
                                        type: "function",
                                        function: { name: event.content_block.name, arguments: "" }
                                    }]
                                },
                                finish_reason: null,
                            }],
                        };
                        res.write(`data: ${JSON.stringify(toolChunk)}\n\n`);
                    }
                } else if (event.type === "content_block_delta") {
                    if (event.delta.type === "text_delta") {
                        const textChunk = {
                            id: "chatcmpl-" + Date.now(),
                            object: "chat.completion.chunk",
                            created: Math.floor(Date.now() / 1000),
                            model: req.body.model || "claude-opus-4-5",
                            choices: [{
                                index: 0,
                                delta: { content: event.delta.text || "" },
                                finish_reason: null,
                            }],
                        };
                        res.write(`data: ${JSON.stringify(textChunk)}\n\n`);
                    } else if (event.delta.type === "input_json_delta") {
                        const toolChunk = {
                            id: "chatcmpl-" + Date.now(),
                            object: "chat.completion.chunk",
                            created: Math.floor(Date.now() / 1000),
                            model: req.body.model || "claude-opus-4-5",
                            choices: [{
                                index: 0,
                                delta: {
                                    tool_calls: [{
                                        index: currentToolCallIndex,
                                        function: { arguments: event.delta.partial_json || "" }
                                    }]
                                },
                                finish_reason: null,
                            }],
                        };
                        res.write(`data: ${JSON.stringify(toolChunk)}\n\n`);
                    }
                } else if (event.type === "content_block_stop") {
                    const idx = event.index;
                    if (contentBlocks[idx]?.type === "tool_use") {
                        currentToolCallIndex++;
                    }
                } else if (event.type === "message_stop") {
                    const stopChunk = {
                        id: "chatcmpl-" + Date.now(),
                        object: "chat.completion.chunk",
                        created: Math.floor(Date.now() / 1000),
                        model: req.body.model || "claude-opus-4-5",
                        choices: [{
                            index: 0,
                            delta: {},
                            finish_reason: currentToolCallIndex > 0 ? "tool_calls" : "stop",
                        }],
                    };
                    res.write(`data: ${JSON.stringify(stopChunk)}\n\n`);
                    res.write("data: [DONE]\n\n");
                }
            } catch (e) {
                console.error("[ERROR] Parse error:", e.message);
            }
        }
    });

    response.data.on("end", () => {
        console.log("[CLAUDE] Stream ended");
        res.end();
    });

    response.data.on("error", (error) => {
        console.error("[ERROR] Stream error:", error.message);
        if (!res.headersSent) {
            res.status(500).json({ error: { message: "Stream error", type: "stream_error" } });
        } else {
            res.end();
        }
    });
}

async function handleGPTRequest(req, res) {
    if (!CONFIG.AZURE_OPENAI_API_KEY) {
        return res.status(500).json({ error: { message: "GPT API key not configured (AZURE_OPENAI_API_KEY)", type: "configuration_error" } });
    }
    if (!CONFIG.AZURE_OPENAI_ENDPOINT) {
        return res.status(500).json({ error: { message: "GPT endpoint not configured (AZURE_OPENAI_ENDPOINT)", type: "configuration_error" } });
    }

    const isStreaming = req.body.stream === true;

    let gptRequest;
    try {
        gptRequest = transformRequestForGPT(req.body);
        console.log("[GPT] Model:", gptRequest.model, "Tools:", gptRequest.tools?.length || 0);
        console.log("[GPT] Input items:", gptRequest.input?.length || 0);
    } catch (transformError) {
        console.error("[ERROR] Transform failed:", transformError.message);
        console.error("[ERROR] Stack:", transformError.stack);
        return res.status(400).json({ error: { message: "Transform error: " + transformError.message, type: "transform_error" } });
    }

    const response = await axios.post(CONFIG.AZURE_OPENAI_ENDPOINT, gptRequest, {
        headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${CONFIG.AZURE_OPENAI_API_KEY}`,
        },
        timeout: 300000,
        responseType: isStreaming ? "stream" : "json",
        validateStatus: (status) => status < 600,
    });

    console.log("[GPT] Response status:", response.status);

    if (response.status >= 400) {
        let errorMessage = "GPT API error";
        if (response.data) {
            if (isStreaming && typeof response.data.pipe === "function") {
                let errorBuffer = "";
                await new Promise((resolve) => {
                    response.data.on("data", (chunk) => { errorBuffer += chunk.toString(); });
                    response.data.on("end", resolve);
                    response.data.on("error", resolve);
                });
                try { errorMessage = JSON.parse(errorBuffer)?.error?.message || errorBuffer; } catch (e) { errorMessage = errorBuffer; }
            } else if (response.data.error) {
                errorMessage = response.data.error.message;
            }
        }
        return res.status(response.status).json({ error: { message: errorMessage, type: "api_error" } });
    }

    if (isStreaming) {
        handleGPTStreaming(req, res, response);
    } else {
        try {
            const openAIResponse = transformGPTResponse(response.data);
            res.json(openAIResponse);
        } catch (transformError) {
            console.error("[ERROR] Response transform failed:", transformError.message);
            return res.status(500).json({ error: { message: "Transform error", type: "transform_error" } });
        }
    }
}

function handleGPTStreaming(req, res, response) {
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");

    let buffer = "";
    let currentToolCallIndex = 0;
    let accumulatedText = "";

    response.data.on("data", (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const data = line.slice(6).trim();
            if (data === "[DONE]") {
                res.write("data: [DONE]\n\n");
                continue;
            }

            try {
                const event = JSON.parse(data);

                // Handle Responses API streaming events
                if (event.type === "response.output_item.added") {
                    // New output item starting
                    if (event.item?.type === "function_call") {
                        const toolChunk = {
                            id: "chatcmpl-" + Date.now(),
                            object: "chat.completion.chunk",
                            created: Math.floor(Date.now() / 1000),
                            model: req.body.model || CONFIG.AZURE_OPENAI_MODEL,
                            choices: [{
                                index: 0,
                                delta: {
                                    tool_calls: [{
                                        index: currentToolCallIndex,
                                        id: event.item.call_id || event.item.id,
                                        type: "function",
                                        function: { name: event.item.name || "", arguments: "" }
                                    }]
                                },
                                finish_reason: null,
                            }],
                        };
                        res.write(`data: ${JSON.stringify(toolChunk)}\n\n`);
                    }
                } else if (event.type === "response.output_text.delta" || event.type === "response.text.delta") {
                    // Text delta
                    const textChunk = {
                        id: "chatcmpl-" + Date.now(),
                        object: "chat.completion.chunk",
                        created: Math.floor(Date.now() / 1000),
                        model: req.body.model || CONFIG.AZURE_OPENAI_MODEL,
                        choices: [{
                            index: 0,
                            delta: { content: event.delta || "" },
                            finish_reason: null,
                        }],
                    };
                    res.write(`data: ${JSON.stringify(textChunk)}\n\n`);
                } else if (event.type === "response.function_call_arguments.delta") {
                    // Function arguments delta
                    const toolChunk = {
                        id: "chatcmpl-" + Date.now(),
                        object: "chat.completion.chunk",
                        created: Math.floor(Date.now() / 1000),
                        model: req.body.model || CONFIG.AZURE_OPENAI_MODEL,
                        choices: [{
                            index: 0,
                            delta: {
                                tool_calls: [{
                                    index: currentToolCallIndex,
                                    function: { arguments: event.delta || "" }
                                }]
                            },
                            finish_reason: null,
                        }],
                    };
                    res.write(`data: ${JSON.stringify(toolChunk)}\n\n`);
                } else if (event.type === "response.output_item.done") {
                    // Output item completed
                    if (event.item?.type === "function_call") {
                        currentToolCallIndex++;
                    }
                } else if (event.type === "response.done" || event.type === "response.completed") {
                    // Response completed
                    const stopChunk = {
                        id: "chatcmpl-" + Date.now(),
                        object: "chat.completion.chunk",
                        created: Math.floor(Date.now() / 1000),
                        model: req.body.model || CONFIG.AZURE_OPENAI_MODEL,
                        choices: [{
                            index: 0,
                            delta: {},
                            finish_reason: currentToolCallIndex > 0 ? "tool_calls" : "stop",
                        }],
                    };
                    res.write(`data: ${JSON.stringify(stopChunk)}\n\n`);
                    res.write("data: [DONE]\n\n");
                } else if (event.type === "content_block_delta" && event.delta?.text) {
                    // Fallback for standard streaming format
                    const textChunk = {
                        id: "chatcmpl-" + Date.now(),
                        object: "chat.completion.chunk",
                        created: Math.floor(Date.now() / 1000),
                        model: req.body.model || CONFIG.AZURE_OPENAI_MODEL,
                        choices: [{
                            index: 0,
                            delta: { content: event.delta.text },
                            finish_reason: null,
                        }],
                    };
                    res.write(`data: ${JSON.stringify(textChunk)}\n\n`);
                }
            } catch (e) {
                console.error("[ERROR] GPT Parse error:", e.message);
            }
        }
    });

    response.data.on("end", () => {
        console.log("[GPT] Stream ended");
        res.end();
    });

    response.data.on("error", (error) => {
        console.error("[ERROR] GPT Stream error:", error.message);
        if (!res.headersSent) {
            res.status(500).json({ error: { message: "Stream error", type: "stream_error" } });
        } else {
            res.end();
        }
    });
}

app.post("/v1/chat/completions", requireAuth, (req, res) => {
    req.url = "/chat/completions";
    app.handle(req, res);
});

app.post("/v1/messages", requireAuth, async (req, res) => {
    console.log("[REQUEST /v1/messages]", new Date().toISOString());
    try {
        if (!CONFIG.AZURE_API_KEY) {
            return res.status(500).json({ error: { message: "Claude API key not configured", type: "configuration_error" } });
        }
        const isStreaming = req.body.stream === true;
        const response = await axios.post(CONFIG.AZURE_ENDPOINT, req.body, {
            headers: {
                "Content-Type": "application/json",
                "x-api-key": CONFIG.AZURE_API_KEY,
                "anthropic-version": req.headers["anthropic-version"] || CONFIG.ANTHROPIC_VERSION,
            },
            timeout: 300000,
            responseType: isStreaming ? "stream" : "json",
        });

        if (isStreaming) {
            res.setHeader("Content-Type", "text/event-stream");
            res.setHeader("Cache-Control", "no-cache");
            res.setHeader("Connection", "keep-alive");
            response.data.pipe(res);
        } else {
            res.json(response.data);
        }
    } catch (error) {
        console.error("[ERROR /v1/messages]", error.message);
        res.status(error.response?.status || 500).json(error.response?.data || { error: { message: error.message, type: "proxy_error" } });
    }
});

app.use((req, res) => {
    res.status(404).json({ error: { message: "Endpoint not found", type: "not_found" } });
});

const server = app.listen(CONFIG.PORT, "0.0.0.0", () => {
    console.log("=".repeat(60));
    console.log("Azure Multi-Model Proxy v3.0 - Claude + GPT");
    console.log("=".repeat(60));
    console.log(`Server: 0.0.0.0:${CONFIG.PORT}`);
    console.log(`Claude: ${CONFIG.AZURE_API_KEY ? "Configured" : "MISSING"}`);
    console.log(`GPT: ${CONFIG.AZURE_OPENAI_API_KEY ? "Configured" : "MISSING"}`);
    console.log(`Endpoints: /chat/completions, /v1/chat/completions, /v1/messages`);
    console.log("=".repeat(60));
});

process.on("SIGTERM", () => { server.close(() => process.exit(0)); });
process.on("SIGINT", () => { server.close(() => process.exit(0)); });
