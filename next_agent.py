"""
Next Question Agent
STT 답변을 분석하여 꼬리질문 또는 질문지의 다음 질문을 선택하는 에이전트

사용법: adk web
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.adk.agents.llm_agent import Agent
from google.adk.tools import FunctionTool, ToolContext
from google.adk.tools.transfer_to_agent_tool import transfer_to_agent
from google.cloud import storage
from pydantic import BaseModel, Field

# --- Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- JSON 저장 디렉토리 설정 (로컬 백업용) ---
JSON_OUTPUT_DIR = Path("json_outputs")
JSON_OUTPUT_DIR.mkdir(exist_ok=True)

logger.info(f"📁 JSON 저장 디렉토리: {JSON_OUTPUT_DIR.absolute()}")

# --- GCS 설정 ---
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "interview-data-cosmic-mariner")
try:
    storage_client = storage.Client()
    logger.info(f"☁️ GCS 버킷: {GCS_BUCKET_NAME}")
except Exception as e:
    logger.warning(f"⚠️ GCS 클라이언트 초기화 실패: {str(e)}")
    storage_client = None


# ===========================
# === LOAD INTERVIEW QUESTIONS ===
# ===========================

def load_from_gcs(filename: str, folder: str = "interview_questions") -> Optional[Dict[str, Any]]:
    """
    GCS에서 JSON 데이터 로드
    
    Args:
        filename: 파일명 (예: "interview_questions_latest.json")
        folder: GCS 내 폴더명 (기본값: "interview_questions")
    
    Returns:
        로드된 데이터 (딕셔너리) 또는 None (실패 시)
    """
    if not storage_client:
        logger.warning("⚠️ GCS 클라이언트가 초기화되지 않아 GCS 로드를 건너뜁니다.")
        return None
    
    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        
        # 최신 파일 찾기 (interview_questions_로 시작하는 파일들)
        if filename == "interview_questions_latest.json":
            blobs = list(bucket.list_blobs(prefix=f"{folder}/interview_questions_"))
            if not blobs:
                logger.warning(f"⚠️ GCS에 질문 파일이 없습니다: gs://{GCS_BUCKET_NAME}/{folder}/")
                return None
            
            # 가장 최근 파일 선택 (이름 기준 정렬 - 타임스탬프 포함)
            blobs.sort(key=lambda x: x.name, reverse=True)
            blob = blobs[0]
            logger.info(f"📥 최신 질문 파일 선택: {blob.name}")
        else:
            blob = bucket.blob(f"{folder}/{filename}")
            
            if not blob.exists():
                logger.warning(f"⚠️ GCS에 파일이 없습니다: gs://{GCS_BUCKET_NAME}/{folder}/{filename}")
                return None
        
        # JSON 데이터 다운로드 및 파싱
        json_string = blob.download_as_text(encoding="utf-8")
        data = json.loads(json_string)
        
        logger.info(f"✅ GCS 로드 완료: gs://{GCS_BUCKET_NAME}/{blob.name}")
        
        return data
        
    except Exception as e:
        logger.error(f"❌ GCS 로드 실패: {str(e)}")
        return None


def load_interview_questions() -> Dict[str, Any]:
    """
    생성된 면접 질문지를 로드합니다.
    우선순위: GCS → 로컬 파일
    
    Returns:
        질문 리스트 및 메타데이터
    """
    
    # 1. GCS에서 최신 질문 파일 로드 시도
    logger.info("🔍 GCS에서 질문지 검색 중...")
    gcs_data = load_from_gcs("interview_questions_latest.json", folder="interview_questions")
    
    if gcs_data:
        logger.info("✅ GCS에서 질문지 로드 성공!")
        data = gcs_data
    else:
        # 2. GCS 실패 시 로컬 파일 사용 (백업)
        logger.info("⚠️ GCS 로드 실패 → 로컬 파일 확인 중...")
        questions_path = JSON_OUTPUT_DIR / "interview_questions.json"
        
        if not questions_path.exists():
            logger.error(f"❌ 질문지를 찾을 수 없습니다: {questions_path}")
            return {
                "error": "질문지가 생성되지 않았습니다. 먼저 interview_agent를 실행하여 질문을 생성해주세요.",
                "questions": []
            }
        
        try:
            with open(questions_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"✅ 로컬 질문지 로드 완료: {questions_path}")
        except Exception as e:
            logger.error(f"❌ 로컬 질문지 로드 오류: {str(e)}")
            return {
                "error": f"질문지 로드 중 오류 발생: {str(e)}",
                "questions": []
            }
    
    # 3. questions를 필요한 형식으로 변환
    questions_list = []
    for idx, q in enumerate(data.get("questions", []), start=1):
        questions_list.append({
            "id": idx,
            "question": q.get("질문 내용", ""),
            "category": q.get("평가하려는 역량", "general"),
            "keywords": q.get("예상 답변 키포인트", []),
            "difficulty": "medium",  # 기본값
            "follow_up_templates": q.get("후속 질문 템플릿", [])
        })
    
    logger.info(f"✅ 질문지 로드 완료: {len(questions_list)}개 질문")
    
    return {
        "company_name": data.get("company_name", ""),
        "resume_summary": data.get("resume_summary", ""),
        "total_questions": len(questions_list),
        "questions": questions_list,
        "source": "GCS" if gcs_data else "Local"
    }


# 질문지 로드 (서버 시작 시)
INTERVIEW_DATA = load_interview_questions()
QUESTION_LIST = INTERVIEW_DATA.get("questions", [])

if INTERVIEW_DATA.get("error"):
    logger.warning(f"⚠️ {INTERVIEW_DATA['error']}")
else:
    source = INTERVIEW_DATA.get("source", "Unknown")
    logger.info(f"📋 [{source}] {INTERVIEW_DATA.get('company_name', '')} 면접 준비 완료!")
    logger.info(f"📝 총 {len(QUESTION_LIST)}개 질문 로드됨")


# --- Interview State (Runtime) ---
INTERVIEW_STATE = {
    "current_question_id": 1,
    "asked_question_ids": [],
    "follow_up_counts": {},  # {question_id: count}
    "interview_context": {
        "asked_questions": [],
        "total_duration": 0
    }
}

logger.info(f"✅ 면접 상태 초기화 완료: 첫 질문 ID={INTERVIEW_STATE['current_question_id']}")


# =============================================================================
# STATE MANAGEMENT FUNCTIONS
# =============================================================================

def update_interview_state(question_id: int, is_follow_up: bool = False):
    """
    면접 상태 업데이트
    
    Args:
        question_id: 질문 ID
        is_follow_up: 꼬리질문 여부
    """
    global INTERVIEW_STATE
    
    if is_follow_up:
        # 꼬리질문 카운트 증가
        current_count = INTERVIEW_STATE["follow_up_counts"].get(question_id, 0)
        INTERVIEW_STATE["follow_up_counts"][question_id] = current_count + 1
        logger.info(f"📝 질문 {question_id}번 꼬리질문 카운트: {current_count + 1}")
    else:
        # 새 질문으로 이동
        if question_id not in INTERVIEW_STATE["asked_question_ids"]:
            INTERVIEW_STATE["asked_question_ids"].append(question_id)
        INTERVIEW_STATE["current_question_id"] = question_id
        INTERVIEW_STATE["follow_up_counts"][question_id] = 0
        logger.info(f"➡️ 질문 {question_id}번으로 이동 (총 {len(INTERVIEW_STATE['asked_question_ids'])}개 질문 완료)")


def get_current_follow_up_count(question_id: int) -> int:
    """현재 질문의 꼬리질문 횟수 반환"""
    return INTERVIEW_STATE["follow_up_counts"].get(question_id, 0)


def get_asked_question_ids() -> List[int]:
    """이미 물어본 질문 ID 리스트 반환"""
    return INTERVIEW_STATE["asked_question_ids"]


# ===========================
# === TOOLS FOR ANSWER EVALUATOR ===
# ===========================

def evaluate_answer(
    question: str,
    answer: str,
    keywords: List[str],
    follow_up_count: int
) -> Dict[str, Any]:
    """
    답변을 평가하여 꼬리질문 필요성 판단
    
    Args:
        question: 현재 질문
        answer: STT로 변환된 답변
        keywords: 질문의 핵심 키워드 리스트
        follow_up_count: 현재 질문에 대한 꼬리질문 횟수
        
    Returns:
        평가 결과 및 꼬리질문 필요성
    """
    
    logger.info(f"📊 답변 평가 시작: {question[:30]}...")
    
    # === 규칙 1: 최대 꼬리질문 횟수 체크 ===
    if follow_up_count >= 2:
        logger.info("⛔ 꼬리질문 2회 도달 → 다음 질문으로")
        return {
            "needs_follow_up": False,
            "reason": "max_follow_ups",
            "analysis": {
                "length": len(answer),
                "keyword_found": False,
                "interest_score": 0
            }
        }
    
    # === 규칙 2: 너무 짧은 답변 ===
    if len(answer) < 20:
        logger.info(f"⚠️ 답변이 너무 짧음 ({len(answer)}자) → 꼬리질문 필요")
        return {
            "needs_follow_up": True,
            "reason": "too_short",
            "analysis": {
                "length": len(answer),
                "keyword_found": False,
                "interest_score": 3
            }
        }
    
    # === 규칙 3: 회피성 답변 ===
    avoid_phrases = ["모르겠", "없습니다", "잘 모릅", "생각나지", "없어요"]
    is_evasive = any(phrase in answer for phrase in avoid_phrases)
    
    if is_evasive:
        logger.info("⚠️ 회피성 답변 감지 → 꼬리질문 필요")
        return {
            "needs_follow_up": True,
            "reason": "evasive",
            "analysis": {
                "length": len(answer),
                "keyword_found": False,
                "interest_score": 2
            }
        }
    
    # === 규칙 4: 키워드 포함 여부 ===
    keyword_found = any(kw.lower() in answer.lower() for kw in keywords)
    
    # === 흥미 포인트 탐지 (간단한 패턴 매칭) ===
    interest_patterns = {
        "numbers": ["배", "배 증가", "명", "등", "위", "%", "명 관리", "팀장", "리더"],
        "conflict": ["갈등", "충돌", "의견 차이", "어려움", "문제", "실패"],
        "achievement": ["달성", "성공", "성과", "개선", "향상", "1등", "수상"],
        "leadership": ["리드", "이끌", "관리", "팀장", "주도"],
        "creative": ["새로운", "창의", "혁신", "아이디어", "방법"]
    }
    
    interest_score = 0
    found_patterns = []
    
    for category, patterns in interest_patterns.items():
        for pattern in patterns:
            if pattern in answer:
                interest_score += 1
                found_patterns.append(category)
                break
    
    logger.info(f"🔍 분석 결과: 길이={len(answer)}, 키워드={keyword_found}, 흥미도={interest_score}")
    
    # === 최종 판단 ===
    # 흥미로운 포인트가 2개 이상 → 꼬리질문으로 파고들기
    if interest_score >= 2:
        logger.info(f"✨ 흥미로운 포인트 발견 ({', '.join(set(found_patterns))}) → 꼬리질문")
        return {
            "needs_follow_up": True,
            "reason": "interesting_point",
            "interest_patterns": list(set(found_patterns)),
            "analysis": {
                "length": len(answer),
                "keyword_found": keyword_found,
                "interest_score": interest_score
            }
        }
    
    # 키워드가 없고 답변이 짧으면 꼬리질문
    if not keyword_found and len(answer) < 100:
        logger.info("⚠️ 키워드 없음 + 짧은 답변 → 꼬리질문")
        return {
            "needs_follow_up": True,
            "reason": "no_keywords_short",
            "analysis": {
                "length": len(answer),
                "keyword_found": False,
                "interest_score": interest_score
            }
        }
    
    # 기본: 충분한 답변
    logger.info("✅ 충분한 답변 → 다음 질문으로")
    return {
        "needs_follow_up": False,
        "reason": "sufficient",
        "analysis": {
            "length": len(answer),
            "keyword_found": keyword_found,
            "interest_score": interest_score
        }
    }


evaluate_answer_tool = FunctionTool(func=evaluate_answer)


def generate_follow_up(
    question: str,
    answer: str,
    reason: str,
    interest_patterns: Optional[List[str]] = None
) -> str:
    """
    꼬리질문 생성
    
    Args:
        question: 원래 질문
        answer: 사용자 답변
        reason: 꼬리질문이 필요한 이유
        interest_patterns: 발견된 흥미 패턴들
        
    Returns:
        생성된 꼬리질문
    """
    
    logger.info(f"💬 꼬리질문 생성: {reason}")
    
    # === 간단한 템플릿 기반 꼬리질문 (빠름) ===
    if reason == "too_short":
        return "조금 더 자세히 설명해주시겠어요?"
    
    elif reason == "evasive":
        return "혹시 관련된 다른 경험이라도 있으신가요?"
    
    elif reason == "no_keywords_short":
        return "구체적인 사례를 들어주실 수 있나요?"
    
    elif reason == "interesting_point":
        # 흥미 패턴에 따른 맞춤 꼬리질문
        if interest_patterns:
            if "numbers" in interest_patterns or "achievement" in interest_patterns:
                questions = [
                    "그 성과를 달성하기 위해 구체적으로 어떤 노력을 하셨나요?",
                    "그 과정에서 가장 어려웠던 점은 무엇이었나요?",
                    "그 결과가 나오기까지 얼마나 걸렸나요?"
                ]
            elif "conflict" in interest_patterns:
                questions = [
                    "그 어려움을 어떻게 극복하셨나요?",
                    "그때 가장 중요하게 생각한 것은 무엇이었나요?",
                    "그 경험을 통해 무엇을 배우셨나요?"
                ]
            elif "leadership" in interest_patterns:
                questions = [
                    "팀을 이끌면서 가장 어려웠던 순간은 언제였나요?",
                    "팀원들과 어떻게 소통하셨나요?",
                    "리더로서 가장 중요하게 생각한 가치는 무엇인가요?"
                ]
            else:
                questions = [
                    "그 경험에서 가장 기억에 남는 순간은?",
                    "구체적으로 어떤 방법을 사용하셨나요?"
                ]
            
            # 첫 번째 질문 반환 (다양성을 위해 나중에 랜덤 선택도 가능)
            return questions[0]
    
    # 기본 꼬리질문
    return "조금 더 구체적으로 말씀해주시겠어요?"


generate_follow_up_tool = FunctionTool(func=generate_follow_up)


# ===========================
# === TOOLS FOR QUESTION MANAGER ===
# ===========================

def check_category_balance(
    asked_questions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    카테고리별 질문 수 확인
    
    Args:
        asked_questions: 이미 물어본 질문 리스트
        
    Returns:
        카테고리별 통계
    """
    
    category_count = {}
    
    for q in asked_questions:
        category = q.get("category", "other")
        category_count[category] = category_count.get(category, 0) + 1
    
    # 목표 분포
    target_distribution = {
        "talent_philosophy": 3,
        "experience": 4,
        "motivation": 2,
        "other": 1
    }
    
    # 부족한 카테고리 찾기
    needs_more = []
    for category, target in target_distribution.items():
        current = category_count.get(category, 0)
        if current < target:
            needs_more.append(category)
    
    logger.info(f"📊 카테고리 현황: {category_count}")
    logger.info(f"⚠️ 부족한 카테고리: {needs_more}")
    
    return {
        "category_count": category_count,
        "target_distribution": target_distribution,
        "needs_more": needs_more
    }


check_category_balance_tool = FunctionTool(func=check_category_balance)


def select_best_question(
    asked_question_ids: List[int],
    category_balance: Dict[str, Any],
    last_answer: str = "",
    question_list: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    최적의 다음 질문 선택
    
    Args:
        asked_question_ids: 이미 물어본 질문 ID 리스트
        category_balance: 카테고리 균형 정보
        last_answer: 마지막 답변 (맥락 연결용)
        question_list: 전체 질문 리스트 (None이면 전역 QUESTION_LIST 사용)
        
    Returns:
        선택된 질문
    """
    
    # 질문 리스트가 제공되지 않으면 전역 QUESTION_LIST 사용
    if question_list is None:
        question_list = QUESTION_LIST
    
    logger.info(f"🎯 다음 질문 선택 시작 (남은 질문: {len(question_list) - len(asked_question_ids)}개)")
    
    # 아직 묻지 않은 질문들만 필터링
    available_questions = [
        q for q in question_list 
        if q.get("id") not in asked_question_ids
    ]
    
    if not available_questions:
        logger.warning("⚠️ 더 이상 질문이 없습니다")
        return {
            "question_id": None,
            "question": "면접을 마무리하겠습니다. 수고하셨습니다!",
            "category": "end",
            "score": 0
        }
    
    # === 점수 계산 ===
    scored_questions = []
    needs_more_categories = category_balance.get("needs_more", [])
    
    for q in available_questions:
        score = 0
        
        # 1. 카테고리 균형 (40점)
        if q.get("category") in needs_more_categories:
            score += 40
        else:
            score += 10
        
        # 2. 맥락 연결 (30점) - 간단한 키워드 매칭
        if last_answer:
            q_keywords = q.get("keywords", [])
            if any(kw.lower() in last_answer.lower() for kw in q_keywords):
                score += 30
            else:
                score += 5
        else:
            score += 15  # 첫 질문은 중간 점수
        
        # 3. 난이도 조절 (20점) - 질문 순서 기반
        total_asked = len(asked_question_ids)
        difficulty = q.get("difficulty", "medium")
        
        if total_asked <= 3:  # 초반 - 쉬운 질문
            if difficulty == "easy":
                score += 20
            elif difficulty == "medium":
                score += 10
        elif total_asked <= 7:  # 중반 - 중간 질문
            if difficulty == "medium":
                score += 20
            else:
                score += 10
        else:  # 후반 - 어려운 질문
            if difficulty == "hard":
                score += 20
            else:
                score += 10
        
        # 4. 시간 효율 (10점) - 중요 카테고리 우선
        important_categories = ["talent_philosophy", "experience", "motivation"]
        if q.get("category") in important_categories:
            score += 10
        else:
            score += 3
        
        scored_questions.append({
            **q,
            "score": score
        })
    
    # 점수 기준 정렬
    scored_questions.sort(key=lambda x: x["score"], reverse=True)
    
    # 최고점 질문 선택
    best_question = scored_questions[0]
    
    logger.info(f"✅ 선택된 질문 (ID: {best_question.get('id')}, 점수: {best_question.get('score')}): {best_question.get('question', '')[:50]}...")
    
    return {
        "question_id": best_question.get("id"),
        "question": best_question.get("question"),
        "category": best_question.get("category"),
        "difficulty": best_question.get("difficulty"),
        "score": best_question.get("score")
    }


select_best_question_tool = FunctionTool(func=select_best_question)


# ===========================
# === AGENTS ===
# ===========================

# --- Sub-Agent 1: Answer Evaluator ---
answer_evaluator_agent = Agent(
    name="answer_evaluator",
    model="gemini-2.5-flash",
    description="면접 답변을 분석하여 꼬리질문 필요성 판단 및 생성",
    instruction="""
당신은 면접 답변 분석 전문가입니다.

**역할:**
1. evaluate_answer 도구로 답변 분석
2. 꼬리질문 필요 시 generate_follow_up 도구로 질문 생성

**분석 기준:**

✅ **꼬리질문이 필요한 경우:**
- 답변이 20자 미만
- 회피성 답변 ("모르겠습니다", "없습니다")
- 흥미로운 포인트 발견 (구체적 수치, 갈등, 성과, 리더십)
- 키워드 없고 답변이 짧음

❌ **다음 질문으로 넘어가는 경우:**
- 충분히 구체적인 답변
- 이미 같은 질문에 꼬리질문 2회 했음

**꼬리질문 스타일:**
- 자연스럽고 대화체
- 긍정적이고 격려하는 톤
- 1문장으로 간결하게

**출력 형식:**
```json
{
  "needs_follow_up": true,
  "reason": "interesting_point",
  "follow_up_question": "그 프로젝트에서 가장 어려웠던 순간은?"
}
```

또는

```json
{
  "needs_follow_up": false,
  "reason": "sufficient"
}
```

반드시 도구를 사용하여 분석하세요.
""",
    tools=[
        evaluate_answer_tool,
        generate_follow_up_tool
    ]
)


# --- Sub-Agent 2: Question Manager ---
question_manager_agent = Agent(
    name="question_manager",
    model="gemini-2.5-flash",
    description="질문지에서 최적의 다음 질문 선택",
    instruction="""
당신은 면접 질문 선택 전문가입니다.

**역할:**
1. check_category_balance로 카테고리 균형 확인
2. select_best_question으로 최적 질문 선택

**선택 전략:**

1. **카테고리 균형** (최우선)
   - 인재상/가치관: 3문
   - 경험/역량: 4문
   - 지원동기: 2문
   - 기타: 1문

2. **맥락 연결**
   - 이전 답변과 자연스럽게 연결되는 질문

3. **난이도 조절**
   - 초반(0-3문): 쉬운 질문
   - 중반(4-7문): 중간 질문
   - 후반(8-10문): 어려운 질문

**출력 형식:**
```json
{
  "question_id": 5,
  "question": "팀 프로젝트에서 갈등을 어떻게 해결하셨나요?",
  "category": "experience",
  "score": 85
}
```

반드시 도구를 사용하여 선택하세요.
""",
    tools=[
        check_category_balance_tool,
        select_best_question_tool
    ]
)


# --- Root Agent: Interview Navigator ---
root_agent = Agent(
    name="interview_navigator",
    model="gemini-2.5-flash",
    description="STT 답변을 분석하여 꼬리질문 또는 질문지의 다음 질문 선택",
    instruction=f"""
당신은 AI 면접관입니다.

**📋 로드된 면접 정보:**
- 기업: {INTERVIEW_DATA.get('company_name', 'Unknown')}
- 총 질문 수: {len(QUESTION_LIST)}개
- 지원자 요약: {INTERVIEW_DATA.get('resume_summary', '')[:100]}...

**🎯 면접 시작 시:**
첫 메시지에서는 다음과 같이 응답하세요:
```
안녕하세요! 똑터뷰 AI 면접관입니다. 😊

{INTERVIEW_DATA.get('company_name', '')} 면접을 시작하겠습니다.
총 {len(QUESTION_LIST)}개의 질문이 준비되어 있습니다.

편안하게 답변해주시면 됩니다.

📌 첫 번째 질문:
{QUESTION_LIST[0].get('question', '') if len(QUESTION_LIST) > 0 else '질문을 불러올 수 없습니다.'}
```

**💬 사용자 답변 수신 시:**

1. **답변 분석**
   - answer_evaluator에게 transfer_to_agent로 전달
   - 평가 결과 받기

2. **의사결정**
   - needs_follow_up = True → 꼬리질문 제시
   - needs_follow_up = False → question_manager에게 다음 질문 요청

3. **질문 출력 형식**

꼬리질문:
```
💬 [꼬리질문]
그 프로젝트에서 가장 어려웠던 순간은 언제였나요?
```

다음 질문:
```
📌 [다음 질문]
LIG Nex1에 지원한 동기를 말씀해주세요.
```

면접 종료:
```
✅ 면접이 모두 완료되었습니다!
수고하셨습니다. 😊
```

**원칙:**
✅ 자연스러운 면접 흐름
✅ 면접자에게 편안한 분위기
✅ 같은 질문에 대한 꼬리질문은 최대 2회
✅ 모든 질문 완료 시 면접 종료 안내

**중요:**
- 첫 메시지가 아닌 경우, 사용자 답변을 answer_evaluator에게 먼저 전달하세요
- 항상 transfer_to_agent를 사용하여 서브 에이전트와 협업하세요
""",
    sub_agents=[
        answer_evaluator_agent,
        question_manager_agent
    ],
    tools=[
        FunctionTool(func=transfer_to_agent)
    ]
)


logger.info("✅ Next Question Agent 준비 완료!")
logger.info("🚀 실행: adk web")
