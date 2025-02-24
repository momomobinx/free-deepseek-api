import json
import requests
import os
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("das_app_client_token_id",
                  default="token")

conversation_id = ""

model_map = {
    "DeepSeek-R1-Distill-Qwen-7B": "DeepSeek-R1-Distill-Qwen-32B",
    "DeepSeek-R1-Distill-Qwen-32B": "DeepSeek-R1-Distill-Qwen-32B",
    "DeepSeek-R1-Distill-Qwen-70B": "DeepSeek-R1-Distill-Qwen-32B",
    "DeepSeek-R1-671B": "DeepSeek-R1-671B",
    "deepseek-reasoner": "DeepSeek-R1-Distill-Qwen-70B"
}


def get_headers():
    global token
    headers = {
        "x-token-dasclient": token
    }
    return headers


def delete_conversation(cookies):
    global conversation_id

    if conversation_id == "":
        return

    # 删除会话
    url = f"https://chat.scnet.cn/api/chat/DeleteMyConversation?conversationId={conversation_id}"
    headers = get_headers()
    requests.post(url=url, headers=headers, cookies=cookies)


async def hunyuan_ds(prompt: str, model: str):
    global token
    global conversation_id

    real_model = model_map[model]

    cookies = {
        "das_app_client_token_id": token
    }

    if prompt.__contains__("删除会话"):
        delete_conversation(cookies)
        result = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1694268190,
            "model": "llava_med_agent",
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "删除会话成功！"
                    },
                    "logprobs": None,
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(result)}\n\n"

        finish = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1694268190,
            "model": "llava_med_agent",
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ]
        }

        yield f"data: {json.dumps(finish)}\n\n"

    ## 发送消息
    url = f"https://chat.scnet.cn/api/chat/Ask"
    headers = get_headers()
    data = {
        "modelType": "DeepSeek-R1-Distill-Qwen-32B",
        "query": prompt,
        "conversationId": conversation_id,
        "messageType": "online"
    }
    ask_response = requests.post(url=url, headers=headers, cookies=cookies, json=data)

    conversation_id = ask_response.json()["data"]["conversation"]["ID"]
    messageId = ask_response.json()["data"]["messageId"]

    url = f"https://chat.scnet.cn/api/chat/GetReplay?messageId={messageId}&query=&modelType={real_model}"
    headers = get_headers()
    response = requests.get(url=url, headers=headers, cookies=cookies, stream=True)
    print(f"status_code: {response.status_code}")
    content_key = "content"
    start = False
    for line in response.iter_lines():
        if line:
            # print(line.decode('utf-8'))
            line_str = line.decode('utf-8')
            if "data: \"\\u003cthink\\u003e\"".__eq__(line_str):
                content_key = "reasoning_content"
                print("思考。。。")
                start = True
                continue
            if "data: \"\\u003c/think\\u003e\"".__eq__(line_str):
                content_key = "content"
                print("回答：")
                continue
            if "event: end".__eq__(line_str):
                start = False
            if not start:
                continue

            msg: str = line_str[len('data: "'):-1]

            msg = msg.replace("\\n", "\n")

            print(msg, end="")
            result = {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1694268190,
                "model": model,
                "system_fingerprint": "fp_44709d6fcb",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            content_key: msg
                        },
                        "logprobs": None,
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(result)}\n\n"

    finish = {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1694268190,
        "model": model,
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [
            {
                "index": 0,
                "delta": {"content": None},
                "logprobs": None,
                "finish_reason": "stop"
            }
        ]
    }
    # yield json.dumps(finish).encode()
    yield f"data: {json.dumps(finish)}\n\n"



# add openai api
from typing import List, Dict, Optional, Union
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI()


# ✅ 定义请求格式
class MessageContent(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None  # 用于图片 URL


class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, Union[str, List[MessageContent]]]]  # content是一个包含多个元素的列表
    stream: Optional[bool] = False  # 是否流式返回
    temperature: float = 0.2
    top_p: float = 0.7
    max_completion_tokens: int = 512


@app.post("/v1/chat/completions")
def chat(request: ChatRequest):
    prompt = request.messages[-1]["content"]
    model = request.model
    gen = hunyuan_ds(prompt, model)

    if request.stream:
        return StreamingResponse(content=gen, media_type="text/event-stream")
    else:
        data = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "pong",
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 1
            }
        }
        return data


# 预定义的模型信息


@app.get("/v1/models")
async def list_models():
    DEEPSEEK_MODELS = {
        "data": [
            {
                "id": "deepseek-reasoner",
                "object": "model",
                "owned_by": "deepseek",
            },
            {
                "id": "DeepSeek-R1-Distill-Qwen-7B",
                "object": "model",
                "owned_by": "deepseek",
            },
            {
                "id": "DeepSeek-R1-Distill-Qwen-32B",
                "object": "model",
                "owned_by": "deepseek",
            },
            {
                "id": "DeepSeek-R1-Distill-Qwen-70B",
                "object": "model",
                "owned_by": "deepseek",
            },
            {
                "id": "DeepSeek-R1-671B",
                "object": "model",
                "owned_by": "deepseek",
            },
        ],
        "object": "list"
    }
    return DEEPSEEK_MODELS


# add openai api end


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
