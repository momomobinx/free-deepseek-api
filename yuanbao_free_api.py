import json
import requests
import os
from dotenv import load_dotenv, set_key

load_dotenv()

hy_token = os.getenv("hy_token",
                     default="your_hy_token")
hy_user = os.getenv("hy_user", default="your_hy_user")
agent_id = "naQivTmsDa"
conversation_id = ""

model_map = {
    "deepseek-chat": "deep_seek_v3",
    "deepseek-reasoner": "deep_seek",
    "hunyuan_gpt_175B_0404": "hunyuan_gpt_175B_0404",
    "hunyuan_t1": "hunyuan_t1"
}


def get_headers(agent_id, content_type):
    headers = {
        "Content-Type": content_type,
        "x-agentid": agent_id,
        "x-requested-with": "XMLHttpRequest",
        "x-source": "web"
    }

    return headers


def update_token(response):
    global hy_token
    global hy_user
    hy_token_new = response.headers.getlist['Set-Cookie']
    hy_user_new = response.cookies["hy_user"]
    if hy_token != hy_token_new:
        set_key(".env", "hy_token", hy_token)
        hy_token = hy_token_new
    if hy_user != hy_user_new:
        set_key(".env", "hy_user", hy_user)
        hy_user = hy_user_new


def delete_conversation(cookies):
    global conversation_id

    if conversation_id == "":
        return

    ## 删除会话
    url = "https://yuanbao.tencent.com/api/user/agent/conversation/v1/clear"
    data = {"conversationIds": [f"{conversation_id}"]}
    content_type = "application/json, text/plain, */*"
    headers = get_headers(agent_id, content_type)
    response = requests.post(url=url, headers=headers, cookies=cookies, json=data)


def create_conversation(cookies) -> str:
    global agent_id
    ## 创建会话
    url = "https://yuanbao.tencent.com/api/user/agent/conversation/create"
    data = {"agentId": agent_id}
    content_type = "application/json, text/plain, */*"
    headers = get_headers(agent_id, content_type)
    response = requests.post(url=url, headers=headers, cookies=cookies, json=data)
    update_token(response)
    new_conversation_id: str = response.json()["id"]
    print(f"创建 conversation_id: {new_conversation_id}")
    return new_conversation_id


async def hunyuan_ds(prompt: str, model: str):
    global hy_user
    global hy_user
    global conversation_id
    global agent_id

    real_model = model_map[model]

    cookies = {
        "hy_token": hy_token,
        "hy_user": hy_user,
        "hy_source": "web"
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
                    "delta": {"content": None},
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ]
        }
        # yield json.dumps(finish).encode()
        yield f"data: {json.dumps(finish)}\n\n"

    if conversation_id == "":
        conversation_id = create_conversation(cookies)

    ## 发送消息
    url = f"https://yuanbao.tencent.com/api/chat/{conversation_id}"
    content_type = "text/plain;charset=UTF-8"
    headers = get_headers(agent_id, content_type)
    data = {"model": "gpt_175B_0404",
            "prompt": prompt,
            "plugin": "Adaptive",
            "displayPrompt": prompt,
            "displayPromptType": 1,
            "options": {
                "imageIntention": {"needIntentionModel": True, "backendUpdateFlag": 2, "intentionStatus": True}},
            "multimedia": [],
            "agentId": agent_id,
            "supportHint": 1,
            "version": "v2",
            "chatModelId": real_model}
    data = json.dumps(data, ensure_ascii=False).encode('utf-8')
    response = requests.post(url=url, headers=headers, cookies=cookies, data=data, stream=True)
    print(f"status_code: {response.status_code}")
    for line in response.iter_lines():
        if line:
            line_str: str = line.decode('utf-8')
            if line_str.__contains__("{"):
                line_json_str = line_str[len('data: '):]
                line_json: map = json.loads(line_json_str)
                content_key = "content"
                msg = ""
                if line_json["type"] == "text":
                    content_key = "content"
                    try:
                        msg = line_json["msg"]
                    except:
                        continue
                elif line_json["type"] == "think":
                    content_key = "reasoning_content"
                    try:
                        msg = line_json["content"]
                    except:
                        continue
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
            else:
                continue

            # yield json.dumps(result).encode()
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
from fastapi import FastAPI, Request

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
async def chat(request: ChatRequest):
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
                "id": "deepseek-chat",
                "object": "model",
                "owned_by": "deepseek",
            },
            {
                "id": "deepseek-reasoner",
                "object": "model",
                "owned_by": "deepseek",
            },
            {
                "id": "hunyuan_gpt",
                "object": "model",
                "owned_by": "tencent",
            },
            {
                "id": "hunyuan_t1",
                "object": "model",
                "owned_by": "tencent",
            },
        ],
        "object": "list"
    }
    return DEEPSEEK_MODELS


# add openai api end


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
