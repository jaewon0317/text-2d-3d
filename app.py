"""
Hunyuan3D 이미지-3D 변환 서버 (로컬 모델 버전)
이 서버는 이미지를 입력받아 3D 모델(GL 형식)로 변환하는 FastAPI 기반 서버입니다.
"""

# ===== 환경 설정 및 라이브러리 임포트 =====
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import time  # [수정] 시간 측정을 위해 time 모듈 임포트
import torch
from PIL import Image
import io
import uuid
from typing import Optional, List
import re

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
from llama_cpp import Llama

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from diffusers import StableDiffusionPipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from langchain_community.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from pydantic import BaseModel


load_dotenv()


# ===== Pydantic 모델 정의 =====
class GemmaSettings(BaseModel):
    temperature: float = 0.7
    max_tokens: int = 512


class Model3DSettings(BaseModel):
    num_inference_steps: int = 5
    octree_resolution: int = 380
    seed: int = 12345


class ChatRequest(BaseModel):
    message: str
    gemma_settings: Optional[GemmaSettings] = None
    model_3d_settings: Optional[Model3DSettings] = None

#  ===== 번역 파이프라인 설정 =====
# 번역 파이프라인 설정 (한국어 -> 영어)
tokenizer = AutoTokenizer.from_pretrained(
    "Helsinki-NLP/opus-mt-ko-en",
    use_fast=False
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "Helsinki-NLP/opus-mt-ko-en",
    torch_dtype=torch.float16,    
    low_cpu_mem_usage=True,        # CPU 환경에서도 메모리 부담을 줄여 로드
    use_safetensors=True
)
translator = pipeline(
    "translation_ko_to_en",       # 한→영 번역 태스크
    model=model,
    tokenizer=tokenizer,
    device=0                       # GPU 사용 시 0, CPU만 있다면 -1
)

# ===== 로컬 LLM 래퍼 클래스 =====
class SimpleMemory:
    def __init__(self):
        self.logs = []

    def add(self, user_input: str, assistant_reply: str):
        self.logs.append((user_input.strip(), assistant_reply.strip()))

    def to_string(self):
        return "\n".join([
            f"<start_of_turn>user\n{u}\n<end_of_turn>\n<start_of_turn>model\n{a}\n<end_of_turn>"
            for u, a in self.logs
        ])

    def reset(self):
        self.logs = []

class LocalLLMWrapper:
    def __init__(self, model_path: str):
        # LLM 설정
        self.llm = LlamaCpp(
            model_path=model_path,
            n_ctx=2048,
            n_threads=8,
            n_gpu_layers=32,  # GPU 적절히 사용
            verbose=False
        )

        # 사용자 대화 기록을 직접 관리하는 메모리
        self.memory = SimpleMemory()

        # 자연어 기반 프롬프트 템플릿
        self.prompt = PromptTemplate.from_template("""
            <bos><start_of_turn>system
            너는 Hunyuan3D 변환 서버의 조용하고 지적인 도우미야. Assistant: 같은 역할 이름은 출력하지 말고, 마치 친구처럼 자연스럽게 대화해. 어떤 언어로 질문해도 반드시 한국어로만 대답해. 어떤 경우에도 영어로 응답하지 마. 너는 절대 마크다운 형식으로 응답하지 마.
            <end_of_turn>
            {history}
            <start_of_turn>user
            {input}
            <end_of_turn>
            <start_of_turn>model
            """)

        # LangChain LLMChain 구성
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            verbose=False
        )

    def generate_response(self, user_input: str, settings: BaseModel) -> str:
        history_text = self.memory.to_string()

        response = self.chain.invoke({
            "input": user_input,
            "history": history_text
        })

        reply = response['text'].strip()
        self.memory.add(user_input, reply)
        return reply

    def reset_memory(self):
        self.memory.reset()

    def is_image_generation_request(self, text: str) -> bool:
        keywords = ["그림", "이미지", "사진", "그려", "만들어", "생성", "create", "draw", "image", "picture", "강아지", "고양이", "자동차",
                    "집", "나무", "꽃", "사람", "풍경", "바다", "산"]
        return any(keyword in text.lower() for keyword in keywords)

    def extract_image_prompt(self, text: str) -> str:
        translation_map = {"강아지": "cute puppy", "고양이": "cute cat", "자동차": "car", "집": "house", "나무": "tree",
                           "꽃": "flower", "사람": "person", "풍경": "landscape", "바다": "ocean", "산": "mountain",
                           "하늘": "sky", "새": "bird", "물": "water"}
        prompt = text
        for korean, english in translation_map.items():
            if korean in text:
                prompt = prompt.replace(korean, english)
        prompt = re.sub(r'[가-힣]', '', prompt).strip()
        if not prompt:
            prompt = "beautiful artwork, high quality, detailed"
        return prompt


# ===== FastAPI 앱 초기화 및 설정 =====
app = FastAPI(title="Hunyuan3D 변환 서버 (로컬 모델)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

LLAMA_MODEL_PATH = os.getenv(
    'LLAMA_MODEL_PATH',
    # 실제 모델 파일(.gguf) 경로를 환경변수로 설정하거나, 아래 기본값을 프로젝트 내 위치로 변경하세요.
    'models/gemma-3-4b-it-q4_0.gguf'
)
try:
    local_llm = LocalLLMWrapper(LLAMA_MODEL_PATH)
    print("로컬 LLM 모델 로드 완료")
except Exception as e:
    print(f"로컬 LLM 모델 로드 실패: {e}")
    exit(1)

HF_TOKEN = os.getenv("HF_TOKEN")
try:
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2',
                                                                subfolder='hunyuan3d-dit-v2-0-turbo',
                                                                use_safetensors=True, device=device)
    pipeline.enable_flashvdm()
    print("Hunyuan3D 모델 로드 완료")
except Exception as e:
    print(f"Hunyuan3D 모델 로드 중 에러 발생: {e}")
    exit(1)

RESULTS_DIR = 'tmp/results'
STATIC_IMG_DIR = Path("static/images")
STATIC_MODEL_DIR = Path("static/3d_models")
os.makedirs(RESULTS_DIR, exist_ok=True)
STATIC_IMG_DIR.mkdir(parents=True, exist_ok=True)
STATIC_MODEL_DIR.mkdir(parents=True, exist_ok=True)

try:
    if HF_TOKEN:
        sd_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", use_auth_token=HF_TOKEN,
                                                          torch_dtype=torch.float16).to(device)
        sd_pipe.enable_attention_slicing()
        print("Stable Diffusion 모델 로드 완료")
    else:
        sd_pipe = None
        print("HF_TOKEN이 없어 Stable Diffusion을 사용할 수 없습니다.")
except Exception as e:
    print(f"Stable Diffusion 로드 실패: {e}")
    sd_pipe = None


# ===== 유틸리티 함수 =====
def process_image(image_data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGBA")
        if image.mode == 'RGB':
            rembg = BackgroundRemover()
            image = rembg(image)
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 처리 중 에러 발생: {str(e)}")


def generate_3d(image: Image.Image, settings: Model3DSettings) -> Path:
    try:
        mesh = pipeline(
            image=image,
            num_inference_steps=settings.num_inference_steps,
            octree_resolution=settings.octree_resolution,
            num_chunks=200000,
            generator=torch.manual_seed(settings.seed),
            output_type='trimesh'
        )[0]
        output_filename = f"{uuid.uuid4()}.glb"
        output_path = os.path.join(RESULTS_DIR, output_filename)
        mesh.export(output_path)
        return Path(output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"3D 변환 중 에러 발생: {str(e)}")


def generate_image_with_sd(prompt: str) -> Optional[Image.Image]:
    if sd_pipe is None: return None
    try:
        return sd_pipe(prompt, guidance_scale=7.5, num_inference_steps=20).images[0]
    except Exception as e:
        print(f"이미지 생성 중 오류: {e}")
        return None


# ===== API 엔드포인트 =====
@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/chat")
async def chat(req: ChatRequest):
    start_time = time.time()  # [수정] 시작 시간 기록

    text = req.message.strip()
    if not text: raise HTTPException(400, "메시지를 입력해 주세요.")

    gemma_settings = req.gemma_settings if req.gemma_settings else GemmaSettings()
    model_3d_settings = req.model_3d_settings if req.model_3d_settings else Model3DSettings()

    try:
        response_text = local_llm.generate_response(text, settings=gemma_settings)
        image_url, model_url = None, None

        if local_llm.is_image_generation_request(text):
            if sd_pipe is not None:
                if re.search(r'[ㄱ-ㅎㅏ-ㅣ가-힣]', text):
                    prompt_en = translator(text, max_length=512)[0]["translation_text"]
                else:
                    prompt_en = text  # 이미 영어니까 그대로
                img_prompt = local_llm.extract_image_prompt(prompt_en)
                print(f"이미지 생성 프롬프트: {img_prompt}")

                img_obj = generate_image_with_sd(img_prompt)
                if img_obj:
                    img_name = f"{uuid.uuid4().hex}.png"
                    img_path = STATIC_IMG_DIR / img_name
                    img_obj.save(img_path)
                    image_url = f"/static/images/{img_name}"

                    try:
                        glb_path = generate_3d(img_obj, settings=model_3d_settings)
                        dest = STATIC_MODEL_DIR / glb_path.name
                        for _ in range(10):
                            try:
                                glb_path.replace(dest);
                                break
                            except PermissionError:
                                time.sleep(0.1)
                        else:
                            raise RuntimeError(f"Failed to move {glb_path} to {dest}")
                        model_url = f"/static/3d_models/{glb_path.name}"
                        response_text += f"\n\n이미지와 3D 모델을 생성했습니다!"
                    except Exception as e:
                        print(f"3D 변환 실패: {e}")
                        response_text += f"\n\n이미지는 생성했지만 3D 변환에 실패했습니다."
                else:
                    response_text += f"\n\n죄송합니다. 이미지 생성에 실패했습니다."
            else:
                response_text += f"\n\n죄송합니다. 이미지 생성 기능이 현재 사용할 수 없습니다."

        duration = time.time() - start_time  # [수정] 종료 시간을 기록하고 소요 시간 계산

        return JSONResponse(
            {"reply": response_text, "image_url": image_url, "model_url": model_url, "duration": duration})
    except Exception as e:
        print(f"Chat 처리 중 오류: {e}")
        raise HTTPException(status_code=500, detail="응답 생성 중 오류가 발생했습니다.")


@app.post("/convert")
async def convert_image(
        file: UploadFile = File(...),
        num_inference_steps: int = Query(5),
        octree_resolution: int = Query(380),
        seed: int = Query(12345)
):
    try:
        settings = Model3DSettings(
            num_inference_steps=num_inference_steps,
            octree_resolution=octree_resolution,
            seed=seed
        )
        contents = await file.read()
        image = process_image(contents)
        start_time = time.time()
        output_path = generate_3d(image, settings)
        elapsed_time = time.time() - start_time
        print(f"변환 완료! 소요 시간: {elapsed_time:.2f}초")
        return FileResponse(output_path, media_type="model/gltf-binary", filename=os.path.basename(output_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    print("로컬 모델 기반 Hunyuan3D 서버를 시작합니다...")
    uvicorn.run(app, host="0.0.0.0", port=8000)