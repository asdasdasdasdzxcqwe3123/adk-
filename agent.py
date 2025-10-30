from __future__ import annotations

import base64
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from google.adk.agents.llm_agent import Agent
from google.adk.tools import FunctionTool, ToolContext
from google.adk.tools.transfer_to_agent_tool import transfer_to_agent
from google.cloud import storage
from pydantic import BaseModel, Field

# --- Dependencies ---
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# --- Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- JSON 저장 디렉토리 설정 ---
JSON_OUTPUT_DIR = Path("json_outputs")
JSON_OUTPUT_DIR.mkdir(exist_ok=True)  # 폴더가 없으면 자동 생성

logger.info(f"📁 JSON 저장 디렉토리: {JSON_OUTPUT_DIR.absolute()}")

# --- GCS 설정 ---
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "interview-data-cosmic-mariner")
try:
    storage_client = storage.Client()
    logger.info(f"☁️ GCS 버킷: {GCS_BUCKET_NAME}")
except Exception as e:
    logger.warning(f"⚠️ GCS 클라이언트 초기화 실패: {str(e)}")
    storage_client = None


# =============================================================================
# GCS 저장 함수
# =============================================================================

def save_to_gcs(
    data: Dict[str, Any], 
    filename: str, 
    folder: str = "interview_questions"
) -> Optional[str]:
    """
    GCS에 JSON 데이터 저장
    
    Args:
        data: 저장할 데이터 (딕셔너리)
        filename: 파일명 (예: "interview_questions_user123_20251027.json")
        folder: GCS 내 폴더명 (기본값: "interview_questions")
    
    Returns:
        GCS URI (gs://interview-data-cosmic-mariner/...) 또는 None (실패 시)
    """
    if not storage_client:
        logger.warning("⚠️ GCS 클라이언트가 초기화되지 않아 GCS 저장을 건너뜁니다.")
        return None
    
    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(f"{folder}/{filename}")
        
        # JSON 데이터를 문자열로 변환하여 업로드
        json_string = json.dumps(data, ensure_ascii=False, indent=2)
        blob.upload_from_string(
            json_string,
            content_type="application/json"
        )
        
        gcs_uri = f"gs://{GCS_BUCKET_NAME}/{folder}/{filename}"
        logger.info(f"✅ GCS 저장 완료: {gcs_uri}")
        
        return gcs_uri
        
    except Exception as e:
        logger.error(f"❌ GCS 저장 실패: {str(e)}")
        return None


# =============================================================================
# TOOLS
# =============================================================================

# === TOOLS ===

# --- Tool 1: Resume Loading ---
class ResumeContentRequest(BaseModel):
    """자기소개서 로딩 요청"""
    pdf_base64: Optional[str] = Field(None, description="Base64 인코딩된 PDF")
    file_path: Optional[str] = Field(None, description="로컬 파일 경로")
    fallback_text: Optional[str] = Field(None, description="직접 입력한 텍스트")

class ResumeContent(BaseModel):
    """자기소개서 내용"""
    resume_text: str = Field(..., description="추출된 텍스트")
    page_count: int = Field(..., description="페이지 수")

def load_resume_content(
    pdf_base64: Optional[str] = None,
    file_path: Optional[str] = None, 
    fallback_text: Optional[str] = None
) -> Dict[str, Any]:
    """자기소개서 PDF 또는 텍스트를 로드합니다.
    
    Args:
        pdf_base64: Base64 인코딩된 PDF 데이터
        file_path: 로컬 PDF 파일 경로
        fallback_text: 직접 입력한 텍스트
        
    Returns:
        Dict containing resume_text and page_count
    """
    
    if pdf_base64:
        if not PdfReader:
            raise RuntimeError("pypdf가 설치되지 않았습니다.")
        try:
            pdf_bytes = base64.b64decode(pdf_base64)
            reader = PdfReader(io.BytesIO(pdf_bytes))
            text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
            if not text:
                raise RuntimeError("PDF에서 텍스트를 추출할 수 없습니다.")
            return {"resume_text": text, "page_count": len(reader.pages)}
        except Exception as e:
            raise RuntimeError(f"PDF 처리 중 오류 발생: {str(e)}")
    
    if file_path:
        if not PdfReader:
            raise RuntimeError("pypdf가 설치되지 않았습니다.")
        try:
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
                if not text:
                    raise RuntimeError("PDF에서 텍스트를 추출할 수 없습니다.")
                return {"resume_text": text, "page_count": len(reader.pages)}
        except Exception as e:
            raise RuntimeError(f"파일 읽기 중 오류 발생: {str(e)}")
    
    if fallback_text:
        text = fallback_text.strip()
        if not text:
            raise RuntimeError("입력된 텍스트가 비어있습니다.")
        return {"resume_text": text, "page_count": 1}
    
    raise RuntimeError("PDF, 파일 경로, 또는 텍스트 중 하나는 필수입니다.")

load_resume_tool = FunctionTool(func=load_resume_content)


# --- Tool 2: Company Research Request (Google Search는 Agent가 직접 수행) ---
def request_company_research(
    company_name: str,
    search_type: str = "overview"
) -> Dict[str, Any]:
    """기업 조사 요청 정보를 반환합니다. 실제 웹 검색은 Agent의 Google Search Grounding이 수행합니다.
    
    Args:
        company_name: 조사할 기업명
        search_type: 조사 유형 (overview, talent_philosophy, core_values, vision, business)
        
    Returns:
        Dict containing research request information
    """
    
    logger.info(f"🔍 기업 조사 요청: {company_name} - {search_type}")
    
    # 검색 가이드 메시지
    research_guides = {
        "overview": f"{company_name}의 전반적인 정보 (인재상, 핵심가치, 비전, 사업분야)",
        "talent_philosophy": f"{company_name}의 인재상과 채용 정보",
        "core_values": f"{company_name}의 핵심 가치와 기업 문화",
        "vision": f"{company_name}의 비전, 미션, 경영 철학",
        "business": f"{company_name}의 주요 사업 분야와 제품/서비스"
    }
    
    guide = research_guides.get(search_type, f"{company_name}에 대한 정보")
    
    return {
        "company_name": company_name,
        "search_type": search_type,
        "research_guide": guide,
        "instruction": f"웹에서 '{guide}'를 검색하여 최신 정보를 수집해주세요. 공식 홈페이지와 신뢰할 수 있는 출처를 우선적으로 참고하세요.",
        "status": "ready_for_search"
    }

company_research_tool = FunctionTool(func=request_company_research)


# --- Tool 2-1: Search Google (더미 도구 - 모델이 명시적으로 검색 수행) ---
def search_google(query: str) -> str:
    """
    Google 검색 도구 - 모델이 이 함수를 호출하여 웹 검색을 수행합니다.
    실제 검색은 Gemini의 내장 Google Search Grounding이 자동으로 수행합니다.
    
    Args:
        query: 검색 쿼리
        
    Returns:
        검색 가이드 메시지
    """
    logger.info(f"🔍 Google 검색 요청: {query}")
    return f"'{query}'에 대한 웹 검색을 수행하세요. Google Search Grounding을 사용하여 최신 정보를 찾고, 공식 출처 URL을 반드시 포함하세요."

search_google_tool = FunctionTool(func=search_google)


# --- Tool 3: Extract Company Information ---
def extract_company_data(
    talent_philosophy: List[str],
    core_values: List[str],
    vision: str,
    business_areas: List[str],
    company_name: str
) -> Dict[str, Any]:
    """Gemini가 웹 검색 결과를 분석한 후 구조화된 기업 정보를 저장합니다.
    
    Args:
        talent_philosophy: 추출한 인재상 리스트
        core_values: 추출한 핵심 가치 리스트
        vision: 추출한 비전/미션
        business_areas: 추출한 사업 분야 리스트
        company_name: 기업명
        
    Returns:
        Dict confirming data was saved
    """
    
    company_data = {
        "company_name": company_name,
        "talent_philosophy": talent_philosophy,
        "core_values": core_values,
        "vision": vision,
        "business_areas": business_areas,
        "timestamp": json.dumps({"note": "Gemini Function Calling으로 추출됨"})
    }
    
    # JSON 파일로 저장 (json_outputs 폴더에)
    output_path = JSON_OUTPUT_DIR / "company_research.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(company_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ 기업 정보 저장 완료: {output_path.absolute()}")
    
    return {
        "status": "success",
        "message": f"{company_name}의 정보가 {output_path}에 저장되었습니다.",
        "data": company_data,
        "file_path": str(output_path)
    }

extract_company_tool = FunctionTool(func=extract_company_data)


# --- Tool 4: Save Interview Questions ---
def save_interview_questions(
    questions: List[Dict[str, Any]],
    resume_summary: str,
    company_name: str
) -> Dict[str, Any]:
    """생성된 면접 질문을 JSON 파일로 저장합니다.
    
    Args:
        questions: 생성된 질문 리스트 (각 질문은 딕셔너리)
        resume_summary: 자기소개서 요약
        company_name: 기업명
        
    Returns:
        Dict confirming questions were saved
    """
    
    # 질문 텍스트만 추출
    questions_text_list = []
    for i, q in enumerate(questions):
        q_text = q.get("질문 내용") or q.get("question") or q.get("text", "")
        if q_text:
            questions_text_list.append(f"{i+1}. {q_text}")
    
    questions_text = "\n".join(questions_text_list)
    
    # 저장할 데이터 구조
    interview_data = {
        "company_name": company_name,
        "resume_summary": resume_summary,
        "total_questions": len(questions),
        "questions": questions,
        "questions_text": questions_text,
        "timestamp": datetime.now().isoformat(),
        "created_by": "interview_agent_adk"
    }
    
    # 파일명 생성 (타임스탬프 포함)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    gcs_filename = f"interview_questions_{timestamp_str}.json"
    
    # 1. 로컬 저장 (개발/디버깅용)
    output_path = JSON_OUTPUT_DIR / "interview_questions.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(interview_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ 로컬 저장 완료: {output_path.absolute()}")
    
    # 2. GCS 저장 (프로덕션용)
    gcs_uri = save_to_gcs(interview_data, gcs_filename, folder="interview_questions")
    
    logger.info(f"📝 총 {len(questions)}개 질문 생성됨")
    
    return {
        "status": "success",
        "message": f"{len(questions)}개의 면접 질문이 저장되었습니다.",
        "local_path": str(output_path),
        "gcs_uri": gcs_uri,
        "question_count": len(questions)
    }

save_questions_tool = FunctionTool(func=save_interview_questions)


# --- Tool 5: Live Interview Session ---
class LiveInterviewRequest(BaseModel):
    """Live 면접 세션 요청"""
    questions: List[str] = Field(..., description="면접 질문 리스트")
    api_key: str = Field(..., description="Google AI API 키")
    
class LiveInterviewResponse(BaseModel):
    """Live 면접 세션 응답"""
    session_id: str = Field(..., description="세션 ID")
    status: str = Field(..., description="세션 상태")
    questions_sent: List[str] = Field(..., description="전송된 질문들")
    message: str = Field(..., description="상태 메시지")

def prepare_live_interview_data() -> Dict[str, Any]:
    """
    Live 면접을 위한 모든 데이터를 준비하고 반환합니다.
    실제 Live Agent는 ADK 외부에서 별도로 처리됩니다.
    """
    
    try:
        # 저장된 질문 데이터 로드 (json_outputs 폴더에서)
        questions_path = JSON_OUTPUT_DIR / "interview_questions.json"
        
        if not questions_path.exists():
            return {
                "status": "error",
                "message": f"면접 질문이 준비되지 않았습니다. {questions_path}에 파일이 없습니다."
            }
        
        with open(questions_path, "r", encoding="utf-8") as f:
            interview_data = json.load(f)
        
        questions = interview_data.get("questions", [])
        questions_text = interview_data.get("questions_text", "")
        resume_summary = interview_data.get("resume_summary", "")
        
        if not questions:
            return {
                "status": "error",
                "message": "면접 질문이 준비되지 않았습니다. 먼저 자기소개서를 업로드하고 질문을 생성해주세요."
            }
        
        # Live 면접을 위한 system_instruction 생성
        live_system_instruction = f"""
🎙️ **똑터뷰 Live 면접관입니다!** 🎙️

안녕하세요! 실시간 음성 면접을 진행하는 AI 면접관입니다.

**📋 자기소개서 분석 결과:**
{resume_summary}

**🎯 이번 면접에서 사용할 맞춤형 질문들:**
{questions_text}

**면접 진행 방식:**
1. 따뜻한 인사와 면접 방식 설명  
2. 위의 맞춤형 질문들을 순서대로 자연스럽게 진행
3. 답변에 따른 적절한 후속 질문 추가
4. 편안하고 격려하는 면접 분위기 유지

면접에 참여해 주셔서 감사합니다! 자기소개서를 검토했고, 맞춤형 질문을 준비했습니다.
편안하게 답변해 주세요. 준비되셨나요? 첫 번째 질문부터 시작하겠습니다!
"""
        
        # Live 면접 데이터 패키지 생성
        live_config = {
            "status": "ready",
            "model": "gemini-live-2.5-flash-preview-native-audio",
            "system_instruction": live_system_instruction,
            "questions": questions,
            "questions_text": questions_text,
            "resume_summary": resume_summary,
            "voice_config": "Kore",
            "response_modalities": ["AUDIO"],
            "enable_affective_dialog": True
        }
        
        # Live 면접 설정 파일로 저장 (json_outputs 폴더에)
        config_path = JSON_OUTPUT_DIR / "live_interview_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(live_config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Live 면접 설정 저장 완료: {config_path.absolute()}")
        
        return {
            "status": "ready",
            "message": f"""
✅ **Live 면접 데이터가 준비되었습니다!** 

**면접 질문 수**: {len(questions)}개
**자기소개서 요약**: {resume_summary[:100]}...

**ADK 시스템의 역할 완료**:
✅ 자기소개서 분석 완료
✅ 맞춤형 질문 생성 완료  
✅ Live 면접 데이터 준비 완료

**저장 위치**: {config_path}

**다음 단계**: 
이제 애플리케이션 레벨에서 Live API를 직접 호출하여 실시간 음성 면접을 시작해야 합니다.

**🚨 중요**: gemini-live 모델은 ADK Agent와 호환되지 않으므로, 
별도의 Live API 클라이언트를 통해 직접 연결해야 합니다.
            """,
            "live_config_file": str(config_path),
            "next_action": "애플리케이션 레벨에서 Live API 직접 호출 필요"
        }
        
    except Exception as e:
        logger.error(f"Live 면접 데이터 준비 실패: {e}")
        return {
            "status": "error", 
            "message": f"Live 면접 데이터 준비 중 오류 발생: {str(e)}"
        }

# 글로벌 변수로 현재 면접 질문들 저장
current_interview_questions = []

prepare_live_data_tool = FunctionTool(func=prepare_live_interview_data)


# === AGENTS ===

# --- Agent 1: Document Analyst ---
document_analyst_agent = Agent(
    name="document_analyst",
    model="gemini-2.5-flash",
    description="자기소개서를 분석하여 핵심 역량과 경험, 지원 기업을 추출하는 전문 에이전트",
    instruction="""
당신은 자기소개서 분석 전문가입니다.

역할:
1. load_resume_content 도구를 사용해 자기소개서 텍스트를 로드하세요
2. 다음 항목들을 분석하여 구조화된 요약을 제공하세요:
   - **지원 기업명** (필수!)
   - 주요 경험 및 프로젝트
   - 핵심 기술 및 역량
   - 성과 및 성취
   - 관심 분야 및 목표
   - 성격 및 가치관

3. 면접에서 깊이 있게 다뤄야 할 포인트들을 식별하세요
4. 잠재적 우려사항이나 확인이 필요한 부분을 표시하세요

5. **중요**: 분석 완료 후 즉시 transfer_to_agent 도구를 사용해서 
   company_researcher 에이전트에게 분석 결과를 전달하세요!

출력 형식: JSON 구조로 각 분석 결과를 정리한 후, 
자동으로 company_researcher로 넘어갑니다.
""",
    tools=[load_resume_tool, FunctionTool(func=transfer_to_agent)],
)

# --- Agent 2: Company Researcher (Google Search Grounding 활성화) ---
company_researcher_agent = Agent(
    name="company_researcher",
    model="gemini-2.5-flash",
    description="Google Search Grounding을 사용하여 웹에서 기업 정보를 실시간으로 검색하고 분석하는 에이전트",
    instruction="""
당신은 기업 조사 전문가입니다. **Google Search를 사용하여 실시간 웹 검색을 수행할 수 있습니다.**

🌐 **중요: 웹 검색 권한**
- 당신은 Google Search에 접근할 수 있습니다
- 최신 정보를 웹에서 직접 검색하여 찾으세요
- 공식 홈페이지와 신뢰할 수 있는 출처를 우선하세요

역할:
1. document_analyst로부터 받은 분석 결과에서 **지원 기업명**을 확인하세요

2. **웹 검색을 수행**하여 다음 정보를 수집하세요:
   
   검색 쿼리 예시:
   - "[기업명] 인재상 채용 공식"
   - "[기업명] 핵심가치 기업문화 공식"
   - "[기업명] 비전 미션 공식"
   - "[기업명] 사업분야 주요사업"

3. 검색 결과에서 다음 정보를 **추출하고 종합**하세요:
   - **인재상**: 기업이 원하는 인재의 특성 (3-5개)
   - **핵심 가치**: 기업의 핵심 가치관 (3-5개)
   - **비전/미션**: 기업의 비전과 미션
   - **사업 분야**: 주요 사업 영역 (2-4개)
   - **⚠️ 중요**: 검색한 출처 URL을 반드시 기록하세요

4. **extract_company_data 함수를 호출**하여 추출한 정보를 저장하세요:
   ```
   extract_company_data(
       talent_philosophy=["창의적이고 도전적인 인재", "협업을 중시하는 인재", "글로벌 마인드를 가진 인재"],
       core_values=["고객 중심", "혁신과 창의", "정직과 신뢰", "상생과 협력"],
       vision="글로벌 선도 기업으로 지속 가능한 성장을 추구합니다",
       business_areas=["반도체", "디스플레이", "모바일"],
       company_name="기업명"
   )
   ```

5. **중요**: 정보 저장 완료 후 transfer_to_agent 도구를 사용하여
   question_generator 에이전트에게 분석 결과와 기업 정보를 전달하세요!

**검색 가이드:**
✅ 공식 홈페이지 우선 참조
✅ 채용 페이지 (careers, recruit) 확인
✅ 기업 소개 페이지 (about, company) 확인
✅ 여러 출처를 종합하여 정확성 확보
✅ 최신 정보 우선 (2024-2025년)

출력: 웹 검색을 통해 수집한 실제 기업 정보를 JSON으로 저장한 후, question_generator로 자동 전달합니다.
""",
    tools=[
        search_google_tool,  # ✅ 명시적 검색 도구 추가
        company_research_tool,
        extract_company_tool,
        FunctionTool(func=transfer_to_agent)
    ],
    # Google Search Grounding 활성화
    # Vertex AI에서 자동으로 grounding 지원 (instruction에서 웹 검색 요청 시)
)

# --- Agent 3: Question Generator ---
question_generator_agent = Agent(
    name="question_generator",
    model="gemini-2.5-flash",
    description="자기소개서 분석과 기업 정보를 바탕으로 맞춤형 면접 질문을 생성하는 에이전트",
    instruction="""
당신은 면접 질문 설계 전문가입니다.

역할:
1. document_analyst의 자기소개서 분석 결과를 받아 해석하세요
2. company_researcher의 기업 정보 조사 결과를 참고하세요 (json_outputs/company_research.json)
3. **기업의 인재상과 핵심 가치를 고려하여** 다음 유형의 질문들을 균형있게 생성하세요:
   - 기본 질문 (자기소개, 지원동기 등) - 2-3개
   - 경험 기반 질문 (구체적 프로젝트, 성과 등) - 3-4개  
   - 행동 면접 질문 (상황별 대응 등) - 2-3개
   - 역량 확인 질문 (기술적/전문적 능력) - 2-3개
   - **기업 적합성 질문 (기업 인재상 연관)** - 2-3개
   - 후속 질문 템플릿 (답변에 따른 추가 질문) - 각 질문당 1-2개

4. 각 질문에 대해 다음 정보를 포함하세요:
   - 질문 내용
   - 질문 의도
   - 평가하려는 역량
   - **기업 인재상과의 연관성**
   - 예상 답변 키포인트
   - 좋은 답변의 기준

5. **중요 - 단계별 순서**:
   ① 먼저 save_interview_questions() 함수를 호출하여 질문을 JSON 파일로 저장하세요
      - questions: 생성한 질문 리스트 (딕셔너리 배열)
      - resume_summary: 자기소개서 요약 (짧게, 2-3문장)
      - company_name: 기업명
   
   ② 그 다음 prepare_live_interview_data() 함수를 호출하여 Live 면접 데이터를 준비하세요

6. 모든 작업 완료 후 이 메시지를 출력하세요:
   "✅ 맞춤형 질문 생성 완료! Live 면접 데이터가 준비되었습니다! 
   이제 애플리케이션 레벨에서 Live API를 직접 호출하여 면접을 시작할 수 있습니다! 🎥"

출력: 최소 12개, 최대 15개의 체계적인 면접 질문 세트를 JSON 형태로 생성하고,
반드시 save_interview_questions 함수를 먼저 호출한 후 prepare_live_interview_data를 호출하세요.
""",
    tools=[save_questions_tool, prepare_live_data_tool],  # 질문 저장 + Live 데이터 준비
)

# --- Agent 3: Live Interviewer (제거됨) ---
# Live 면접은 ADK 외부에서 별도의 Live API 클라이언트로 처리됩니다.
# ADK는 면접 준비(분석, 질문 생성, 데이터 준비)까지만 담당합니다.

# --- Agent 4: Feedback Generator ---
feedback_generator_agent = Agent(
    name="feedback_generator",
    model="gemini-2.0-flash-exp", 
    description="면접 결과를 종합하여 점수와 피드백을 생성하는 에이전트",
    instruction="""
당신은 면접 평가 및 피드백 전문가입니다.

역할:
1. 면접 전 과정의 데이터를 종합 분석하세요:
   - 자기소개서 분석 결과
   - 사용된 질문들
   - 면접 진행 로그 (가능한 경우)

2. 다음 기준으로 평가하세요:
   - 의사소통 능력 (25%)
   - 전문성 및 경험 (30%) 
   - 문제해결 능력 (20%)
   - 열정 및 적합성 (15%)
   - 성장 가능성 (10%)

3. 피드백 리포트를 생성하세요:
   - 총점 (100점 만점)
   - 항목별 세부 점수
   - 주요 강점 (3-5개)
   - 개선이 필요한 영역 (3-5개)
   - 구체적인 발전 방안
   - 추천 학습 자료 또는 준비 방법

출력: 후보자가 이해하기 쉽고 실행 가능한 피드백 리포트를 JSON과 자연어로 제공하세요.
""",
)

# --- Root Agent (Orchestrator) ---
root_agent = Agent(
    name="multi_agent_interview_system",
    model="gemini-2.5-flash",
    description="AI 면접 시스템을 총괄 관리하는 코디네이터 에이전트 (Gemini Function Calling 기반)",
    instruction="""
🎯 **똑터뷰 AI 면접 시스템에 오신 것을 환영합니다!** 🎯

저는 자기소개서 기반 맞춤형 면접을 총괄 관리하는 코디네이터입니다.

**시스템 작동 방식:**

📝 **면접 준비 단계 (ADK + Gemini Function Calling 자동화)**
1. 자기소개서 업로드 → document_analyst에게 분석 위임 (기업명 추출)
2. 기업명 추출 → company_researcher에게 기업 조사 위임
   🔍 **Gemini가 Function Calling으로 웹 검색 자동 수행!**
   - search_company_info() 함수를 Gemini가 직접 호출
   - 인재상, 핵심가치, 비전, 사업분야 등 자동 수집
   - extract_company_data() 함수로 JSON 저장
3. 분석 + 기업 정보 → question_generator에게 맞춤형 질문 생성 위임
4. 질문 생성 완료 → Live 면접 데이터 자동 준비 (live_interview_config.json 생성)

🎥 **Live 면접 단계 (애플리케이션 레벨 직접 처리)**
- ADK 시스템이 준비한 Live 면접 데이터를 사용
- 애플리케이션에서 Live API를 직접 호출하여 실시간 음성 면접 진행
- 면접 완료 후 다시 ADK로 복귀 → feedback_generator에게 평가 위임

**🚨 핵심 기능: Gemini Function Calling**
Gemini가 필요할 때마다 자동으로 웹 검색 함수를 호출합니다:
- search_company_info(company_name, search_type) 
- extract_company_data(talent_philosophy, core_values, ...)

**면접 준비 상태 확인:**
- 자기소개서 분석: ❌ (업로드 필요)
- 기업 정보 조사: ❌ (Gemini Function Calling으로 자동 수행)
- 맞춤형 질문 생성: ❌ (기업 정보 반영하여 자동 생성)
- Live 면접 데이터 준비: ❌ (질문 생성 후 자동 준비)

**자동화된 프로세스 (ADK 담당):**
자기소개서만 업로드하시면 면접 준비까지 모두 자동으로 진행됩니다!
1. 📄 업로드 → 🔍 자동 분석 (기업명 추출)
2. 🌐 Gemini Function Calling으로 웹 검색 (인재상, 핵심가치 등)
3. ❓ 기업 정보를 반영한 맞춤형 질문 생성
4. 📋 Live 데이터 준비

**출력 파일 (json_outputs/ 폴더에 저장):**
- evaluate_polio.json: 자기소개서 분석 결과
- company_research.json: 🆕 기업 조사 결과 (Google Search Grounding)
- interview_questions.json: 생성된 면접 질문
- live_interview_config.json: Live 면접 설정

**현재 단계:**
자기소개서를 업로드해 주시면 Gemini가 Function Calling으로 웹을 검색하여
기업 정보를 수집하고, 맞춤형 질문을 생성합니다!

🔧 **아키텍처**: ADK(준비) + Gemini Function Calling(웹 검색) + Live API(면접)

시작할 준비가 되셨나요? 📄✨
""",
    sub_agents=[
        document_analyst_agent,
        company_researcher_agent,
        question_generator_agent,
        feedback_generator_agent,
    ],
    tools=[FunctionTool(func=transfer_to_agent)],
    include_contents="default",
)
