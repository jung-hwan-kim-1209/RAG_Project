"""
Layer 10: OUTPUT LAYER
최종 출력을 처리하는 레이어
"""
import json
import csv
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path

from models import InvestmentReport, QualityCheckResult, PipelineContext
from layers.report_generation_layer import ReportFormatter

class OutputProcessor:
    """출력 처리기"""

    def __init__(self):
        self.formatter = ReportFormatter()

    def format_console_output(self, context: PipelineContext) -> str:
        """콘솔 출력 포맷팅"""
        if not context.final_report:
            return "❌ 투자 평가 리포트 생성 실패"

        output_parts = []
        
        # 1. 메인 리포트
        formatted_report = self.formatter.format_console_report(context.final_report)
        output_parts.append(formatted_report)

        # 2. 품질 검증 결과
        quality_section = self._format_quality_section(context.quality_check)
        if quality_section:
            output_parts.append(quality_section)

        # 3. 데이터 출처 정보
        if context.final_report.data_sources:
            sources_section = self._format_data_sources(context.final_report.data_sources)
            output_parts.append(sources_section)

        # 4. 처리 과정 (옵션)
        if context.processing_steps and len(context.processing_steps) > 0:
            steps_section = self._format_processing_steps(context.processing_steps)
            output_parts.append(steps_section)

        return "\n\n".join(output_parts)

    def _format_quality_section(self, quality_check: Optional[QualityCheckResult]) -> str:
        """품질 검증 섹션 포맷팅"""
        if not quality_check:
            return ""

        lines = ["📋 품질 검증 결과"]
        lines.append("─" * 50)

        if quality_check.passed:
            lines.append(f"✅ 품질 검증 통과 (품질 점수: {quality_check.overall_quality:.1%})")
        else:
            lines.append(f"⚠️ 품질 검증 실패")
            lines.append(f"   발견된 이슈: {len(quality_check.issues)}개")

            # 주요 이슈 표시 (상위 3개)
            if quality_check.issues:
                lines.append("\n   주요 이슈:")
                for i, issue in enumerate(quality_check.issues[:3], 1):
                    severity, message = self._resolve_issue_fields(issue)
                    lines.append(f"   {i}. [{severity}] {message}")

        return "\n".join(lines)

    @staticmethod
    def _resolve_issue_fields(issue: Any) -> Tuple[str, str]:
        """이슈 객체에서 표시할 필드를 안전하게 추출"""
        default_message = str(issue)

        if issue is None:
            return ("정보", "")

        # dataclass / 객체 형태 처리
        severity = getattr(issue, "severity", None)
        message = getattr(issue, "message", None)

        if severity and message:
            return (severity, message)

        # dict 형태 지원
        if isinstance(issue, dict):
            return (
                str(issue.get("severity", "정보")),
                str(issue.get("message", issue.get("detail", default_message)))
            )

        return (severity or "정보", message or default_message)

    def _format_data_sources(self, data_sources: List[str]) -> str:
        """데이터 출처 포맷팅"""
        lines = ["📚 데이터 출처"]
        lines.append("─" * 50)
        for i, source in enumerate(data_sources[:5], 1):  # 최대 5개만 표시
            lines.append(f"{i}. {source}")
        
        if len(data_sources) > 5:
            lines.append(f"   ... 외 {len(data_sources) - 5}개")
        
        return "\n".join(lines)

    def _format_processing_steps(self, steps: List[str]) -> str:
        """처리 과정 포맷팅"""
        lines = ["🔄 처리 과정"]
        lines.append("─" * 50)
        for i, step in enumerate(steps, 1):
            lines.append(f"{i:2d}. {step}")
        return "\n".join(lines)

    def save_report_to_file(
        self,
        report: InvestmentReport,
        output_path: str,
        format_type: str = "json"
    ) -> bool:
        """리포트를 파일로 저장"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            if format_type == "json":
                json_data = self.formatter.format_json_report(report)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)

            elif format_type == "txt":
                text_report = self.formatter.format_console_report(report)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(text_report)

            elif format_type == "csv":
                self._save_as_csv(report, output_file)

            else:
                raise ValueError(f"지원하지 않는 출력 형식: {format_type}")

            print(f"✅ 리포트 저장 완료: {output_file}")
            return True

        except Exception as e:
            print(f"❌ 파일 저장 실패: {e}")
            return False

    def _save_as_csv(self, report: InvestmentReport, output_file: Path) -> None:
        """CSV 형식으로 저장"""
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # 헤더
            writer.writerow(['항목', '값', '비고'])

            # 기본 정보
            writer.writerow(['회사명', report.company_info.name, ''])
            writer.writerow(['업종', report.company_info.industry, ''])
            writer.writerow(['평가일', report.evaluation_date.strftime('%Y-%m-%d'), ''])
            writer.writerow(['', '', ''])  # 빈 줄

            # 점수 정보
            writer.writerow(['총점', report.unicorn_score.total_score, '100점 만점'])
            writer.writerow(['등급', report.unicorn_score.grade, ''])
            writer.writerow(['유니콘확률', f"{report.unicorn_score.unicorn_probability:.1%}", ''])
            writer.writerow(['투자추천', report.recommendation.value, ''])
            writer.writerow(['신뢰도', f"{report.confidence_level:.1%}", ''])
            writer.writerow(['', '', ''])  # 빈 줄

            # 영역별 점수
            writer.writerow(['분석영역', '점수', '등급'])
            for result in report.analysis_results:
                writer.writerow([result.category, result.score, result.grade])

    def export_summary_metrics(self, report: InvestmentReport) -> Dict[str, Any]:
        """요약 지표 추출"""
        return {
            "company_name": report.company_info.name,
            "evaluation_date": report.evaluation_date.isoformat(),
            "total_score": report.unicorn_score.total_score,
            "grade": report.unicorn_score.grade,
            "unicorn_probability": report.unicorn_score.unicorn_probability,
            "recommendation": report.recommendation.value,
            "confidence_level": report.confidence_level,
            "analysis_results_summary": {
                "total_categories": len(report.analysis_results),
                "average_score": sum(r.score for r in report.analysis_results) / len(report.analysis_results) if report.analysis_results else 0,
                "high_scores": len([r for r in report.analysis_results if r.score >= 80])
            },
            "risk_assessment_summary": {
                "total_risks": len(report.risk_assessments),
                "high_risks": len([r for r in report.risk_assessments if r.risk_level.value in ["높음", "매우 높음"]]),
                "critical_risks": len([r for r in report.risk_assessments if r.risk_level.value == "매우 높음"])
            },
            "data_sources_count": len(report.data_sources)
        }


class OutputLayer:
    """출력 레이어 메인 클래스"""

    def __init__(self):
        self.output_processor = OutputProcessor()

    def process_final_output(
        self,
        context: PipelineContext,
        output_format: str = "console",
        save_to_file: bool = False,
        output_path: str = None
    ) -> str:
        """최종 출력 처리"""

        # 실행 완료 시간 기록
        context.execution_end_time = datetime.now()
        
        # processing_steps에 완료 기록
        if context.execution_start_time and context.execution_end_time:
            duration = context.execution_end_time - context.execution_start_time
            context.processing_steps.append(
                f"Output Layer: 최종 출력 생성 완료 (소요시간: {duration.total_seconds():.1f}초)"
            )

        if output_format == "console":
            output = self.output_processor.format_console_output(context)

            # 파일 저장 옵션
            if save_to_file and output_path and context.final_report:
                file_format = self._determine_output_format(output_format, output_path)
                self.output_processor.save_report_to_file(
                    context.final_report,
                    output_path,
                    file_format
                )

            return output

        elif output_format == "json":
            if context.final_report:
                json_data = self.output_processor.formatter.format_json_report(context.final_report)
                return json.dumps(json_data, ensure_ascii=False, indent=2)
            else:
                return json.dumps({"error": "리포트 생성 실패"}, ensure_ascii=False)

        elif output_format == "summary":
            if context.final_report:
                summary = self.output_processor.export_summary_metrics(context.final_report)
                return json.dumps(summary, ensure_ascii=False, indent=2)
            else:
                return json.dumps({"error": "리포트 생성 실패"}, ensure_ascii=False)

        else:
            return self.output_processor.format_console_output(context)

    @staticmethod
    def _determine_output_format(output_format: str, output_path: Optional[str]) -> str:
        """저장 시 사용할 파일 포맷 결정"""
        valid_formats = {"json", "txt", "csv"}

        if output_format in valid_formats:
            return output_format

        if output_path:
            suffix = Path(output_path).suffix.lower()
            mapping = {".json": "json", ".txt": "txt", ".csv": "csv"}
            if suffix in mapping:
                return mapping[suffix]

        return "json"

    def print_processing_summary(self, context: PipelineContext) -> None:
        """처리 과정 요약 출력"""
        print("\n" + "="*60)
        print("🔄 파이프라인 실행 요약")
        print("="*60)

        if context.execution_start_time and context.execution_end_time:
            duration = context.execution_end_time - context.execution_start_time
            print(f"⏱️  총 실행 시간: {duration.total_seconds():.2f}초")

        print(f"📊 처리된 단계: {len(context.processing_steps)}개")
        print()

        for i, step in enumerate(context.processing_steps, 1):
            print(f"  {i:2d}. {step}")

        if context.quality_check:
            print(f"\n✨ 최종 품질 점수: {context.quality_check.overall_quality:.1%}")
            if not context.quality_check.passed:
                print(f"⚠️  품질 이슈: {len(context.quality_check.issues)}개 발견")

        print("="*60 + "\n")


def create_output_layer() -> OutputLayer:
    """Output Layer 생성자"""
    return OutputLayer()


def process_output_layer(
    context: PipelineContext,
    output_format: str = "console",
    save_to_file: bool = False,
    output_path: str = None,
    show_summary: bool = True
) -> str:
    """Output Layer 처리 함수"""
    output_layer = create_output_layer()

    # 최종 출력 생성
    final_output = output_layer.process_final_output(
        context=context,
        output_format=output_format,
        save_to_file=save_to_file,
        output_path=output_path
    )

    # 처리 과정 요약 (콘솔 모드에서만, 옵션)
    if output_format == "console" and show_summary:
        output_layer.print_processing_summary(context)

    return final_output
