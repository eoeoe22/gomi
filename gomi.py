#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
무한 개소리 생성기 (Infinite Nonsense Generator)
------------------------------------------------
동작 원리:
  1) Gemma 사전학습(base, -pt) 모델을 로드. (instruct 아님 = 문장 이어붙이기만 함)
  2) 랜덤 단어를 마구 섞어 "문장 아닌 문장"을 시드로 입력.
  3) 높은 temperature 로 출력.
  4) 출력에서 단어 몇 개를 랜덤으로 뽑아 다음 입력으로 사용.
     -> 이전 컨텍스트(출력 전체)는 버린다. 매 라운드가 독립.
  5) 무한 반복 (Ctrl+C 로 종료)

하드웨어: RTX 5070 Laptop / VRAM 8GB
  - 1b-pt: bf16 약 2GB  (여유로움, 기본값)
  - 2b   : bf16 약 5GB  (가능)
  - 4b-pt: 4bit 약 3GB  (bitsandbytes 필요, 아래 USE_4BIT 참고)
"""

import os
import sys
import re
import time
import random
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

try:
    import msvcrt
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False

class NewSeedRequested(Exception):
    """ESC 키가 눌려 새로운 시드 입력을 요청할 때 발생하는 예외"""
    pass


# ============================ 설정 ============================
MODEL_ID   = os.environ.get("MODEL_ID", "google/gemma-3-1b-pt")  # 반드시 base(-pt)!
USE_4BIT   = False     # True 로 켜면 더 큰 모델(예: gemma-3-4b-pt)을 8GB에 욱여넣기

SEED_WORDS     = 7     # 시작 시드에 쓸 단어 수
FEEDBACK_WORDS = 4     # 출력에서 다음 입력으로 재활용할 단어 수
MAX_NEW_TOKENS = 256   # 한 라운드에 생성할 토큰 수 (기존 64에서 256으로 상향)

# 동적 온도 조절 (Dynamic Temperature) 설정
USE_DYNAMIC_TEMP    = True   # True: 루프마다 온도를 사인파로 변동시킴, False: 고정 온도 사용
DYNAMIC_TEMP_MIN    = 1.2    # 최소 온도 (상대적으로 정상적인 출력)
DYNAMIC_TEMP_MAX    = 2.0    # 최대 온도 (광기 어린 무작위 출력)
DYNAMIC_TEMP_PERIOD = 10     # 사인파의 한 주기(주기당 라운드 수)
SHOW_TEMP_INDICATOR = False  # 생성 시작 시 현재 온도를 화면에 표시할지 여부 (이제 타이틀바에 표시되므로 False 권장)

TEMPERATURE        = 1.7   # USE_DYNAMIC_TEMP가 False일 때 사용할 고정 온도 (1.3 ~ 2.0 추천)
TOP_P              = 0.97
TOP_K              = 0     # 0 = 비활성(완전 카오스). 출력이 깨지면 80~120 으로
REPETITION_PENALTY = 1.15  # 같은 단어 무한반복 방지

SLEEP_BETWEEN = 0.4   # 라운드 사이 쉬는 시간(초). 0 이면 쉬지 않음
# =============================================================


# ---------- 랜덤 단어 풀 만들기 ----------
DEFAULT_WORDS = """
banana velvet thunder marble whisper engine pickle galaxy ribbon cactus
mirror plastic comet jelly anchor lantern fossil noodle pebble syrup
quantum carpet dragon biscuit magnet violin orbit pretzel feather kettle
hollow trumpet glacier muffin spiral walnut cobweb turbine pixel mango
shadow bucket crayon tundra zipper lemon goblin yacht wrinkle tofu
허무 바나나 구름 망치 고양이 우주 양말 라면 번개 거울
""".split()

def load_wordbank():
    """시스템 사전이 있으면 그걸 쓰고, 없으면 기본 단어 풀 사용."""
    for path in ("/usr/share/dict/words",):
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    w = [ln.strip() for ln in f
                         if ln.strip().isalpha() and 3 <= len(ln.strip()) <= 11]
                if len(w) > 500:
                    return w
            except Exception:
                pass
    return DEFAULT_WORDS

WORDBANK = load_wordbank()

def random_seed(n):
    """랜덤 단어 n개를 섞어 '문장 아닌 문장'을 만든다."""
    return " ".join(random.sample(WORDBANK, k=min(n, len(WORDBANK))))

def pick_words(text, n):
    """출력 텍스트에서 단어 n개를 랜덤으로 뽑는다. 비면 새 시드로 폴백."""
    words = re.findall(r"\w+", text)          # 유니코드 단어(한/영/숫자) 매칭
    if not words:
        return random_seed(n)
    random.shuffle(words)
    return " ".join(words[:n])


# ---------- 모델 로드 ----------
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️  CUDA를 못 찾았습니다. CPU로 돌아가지만 매우 느립니다.")
        print("    RTX 50 시리즈면 cu128 빌드 PyTorch가 설치됐는지 확인하세요.")

    set_terminal_title(f"⏳ 모델 로딩 중: {MODEL_ID} ...")
    print(f"모델 로딩 중: {MODEL_ID}  (device={device}) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    kwargs = dict(torch_dtype=torch.bfloat16)
    if USE_4BIT:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["device_map"] = "cuda"

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    if not USE_4BIT:
        model.to(device)
    model.eval()
    print("로딩 완료. 개소리를 시작합니다. (Ctrl+C 로 종료)\n")
    return tokenizer, model


def set_terminal_title(title):
    """터미널 창의 제목을 설정합니다."""
    try:
        if os.name == 'nt':
            import ctypes
            ctypes.windll.kernel32.SetConsoleTitleW(title)
        else:
            sys.stdout.write(f"\033]0;{title}\a")
            sys.stdout.flush()
    except Exception:
        pass


def check_pause():
    """키 입력이 감지되면 일시정지하거나, ESC의 경우 시드 재입력 예외를 발생시킵니다."""
    if HAS_MSVCRT and msvcrt.kbhit():
        ch = msvcrt.getch()
        if ch == b'\x1b':  # ESC key
            raise NewSeedRequested()
        
        # 다른 키인 경우 일시정지
        # 기존 버퍼 비우기
        while msvcrt.kbhit():
            msvcrt.getch()
        print("\n\n[일시정지 - 아무 키나 누르면 재개합니다...]", end="", flush=True)
        # 키 입력 대기
        msvcrt.getch()
        # 재개 시 누른 키 버퍼 비우기
        while msvcrt.kbhit():
            msvcrt.getch()
        print("[재개]\n", end="", flush=True)


class NoNewlineStreamer(TextStreamer):
    """줄바꿈을 모두 공백으로 대체하여 한 줄로 계속 흐르게 하는 스트리머"""
    def on_finalized_text(self, text: str, stream_end: bool = False):
        # 모든 줄바꿈 문자를 공백으로 변경
        clean_text = text.replace("\n", " ").replace("\r", "")
        print(clean_text, end="", flush=True)
        
        # 스트리밍 도중 키 입력 감지 시 일시정지
        check_pause()
        
        if stream_end:
            # 라운드가 끝날 때 줄바꿈 대신 공백 하나를 출력하여 이어지게 함
            print(" ", end="", flush=True)


# ---------- 한 라운드 생성 ----------
def generate(tokenizer, model, prompt, temp=TEMPERATURE):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = NoNewlineStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    if SHOW_TEMP_INDICATOR:
        # 온도 수준에 맞춰 색상 설정 (Green -> Yellow -> Red)
        if temp < 1.4:
            color = "\033[92m"  # Green
        elif temp < 1.7:
            color = "\033[93m"  # Yellow
        else:
            color = "\033[91m"  # Red
        print(f"{color}[🌡️ {temp:.2f}]\033[0m ", end="", flush=True)

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=temp,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        streamer=streamer,
        pad_token_id=tokenizer.eos_token_id,
    )
    if TOP_K > 0:
        gen_kwargs["top_k"] = TOP_K

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    # 새로 생성된 토큰만 잘라서 디코드 (다음 라운드 시드 추출용)
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------- 메인 루프 ----------
def main():
    set_terminal_title("⏳ 모델 로드 대기 중...")
    tokenizer, model = load_model()
    set_terminal_title("✨ 준비 완료")
    
    def get_user_seed():
        if HAS_MSVCRT:
            while msvcrt.kbhit():
                msvcrt.getch()
        user_input = input("\n시작 시드 문장을 입력하세요 (엔터 입력 시 랜덤 생성): ").strip()
        if user_input:
            print()
            return user_input
        else:
            seed = random_seed(SEED_WORDS)
            print(f"랜덤 시드 사용: {seed}\n")
            return seed

    prompt = get_user_seed()

    try:
        step = 0
        while True:
            try:
                if USE_DYNAMIC_TEMP:
                    # 1.2에서 2.0 사이로 코사인/사인파 변동 (step=0일 때 최소 온도 1.2에서 시작하여 파도를 치며 상승)
                    theta = 2 * math.pi * step / DYNAMIC_TEMP_PERIOD
                    temp = DYNAMIC_TEMP_MIN + (DYNAMIC_TEMP_MAX - DYNAMIC_TEMP_MIN) * (1 - math.cos(theta)) / 2
                else:
                    temp = TEMPERATURE

                # 터미널 제목 표시줄에 온도와 단계 정보 표시
                set_terminal_title(f"🌡️ Temp: {temp:.2f} | Step: {step} | 무한 개소리 생성기")

                output = generate(tokenizer, model, prompt, temp=temp)   # 토큰이 실시간 출력됨
                prompt = pick_words(output, FEEDBACK_WORDS)    # 출력 → 다음 시드
                step += 1

                if SLEEP_BETWEEN:
                    if HAS_MSVCRT:
                        # 슬립 중에도 실시간 키 감지하여 일시정지 처리
                        start_time = time.time()
                        while time.time() - start_time < SLEEP_BETWEEN:
                            if msvcrt.kbhit():
                                check_pause()
                                break
                            time.sleep(0.05)
                    else:
                        time.sleep(SLEEP_BETWEEN)
            except NewSeedRequested:
                print("\n\n[🛑 ESC 감지: 생성을 중단하고 새 시드를 입력받습니다.]")
                set_terminal_title("🛑 새 시드 대기 중...")
                prompt = get_user_seed()
                step = 0  # 단계 및 온도 곡선 초기화
    except KeyboardInterrupt:
        print("\n\n[종료] 개소리 생성을 멈춥니다. 수고하셨습니다.")
        set_terminal_title("무한 개소리 생성기 (종료됨)")


if __name__ == "__main__":
    main()