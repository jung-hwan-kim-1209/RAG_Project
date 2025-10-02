"""
Layer 10: OUTPUT LAYER
최종 출력을 처리하는 레이어
"""
import json
import csv
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import InvestmentReport, QualityCheckResult, PipelineContext
from layers.report_generation_layer import ReportFormatter

class OutputProcessor:
    """출력 처리기"""

    def __init__(self):
        self.formatter = ReportFormatter()

    def format_console_output(self, context: PipelineContext) -> str:
        """콘솔 출력 포맷팅"""
        if not context.final_report:
            return "[오류] 투자 평가 리포트 생성 실패"

        # 품질 검증 상태 확인
        quality_status = ""
        if context.quality_check:
            if context.quality_check.passed:
                quality_status = "[완료] 품질 검증 통과"
            else:
                quality_status = f"[경고] 품질 검증 실패 (이슈: {len(context.quality_check.issues)}개)"

        # 기본 리포트 포맷팅
        formatted_report = self.formatter.format_console_report(context.final_report)

        # 품질 정보 추가
        if quality_status:
            formatted_report += f"\n\n{quality_status}\n"

        # 처리 과정 요약 추가
        if context.processing_steps:
            formatted_report += "\n[처리] 처리 과정:\n"
            for i, step in enumerate(context.processing_steps, 1):
                formatted_report += f"{i}. {step}\n"

        return formatted_report

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

            elif format_type == "pdf":
                return self.formatter.format_pdf_report(report, str(output_file))

            return True

        except Exception as e:
            print(f"파일 저장 실패: {e}")
            return False

    def _save_as_csv(self, report: InvestmentReport, output_file: Path) -> None:
        """CSV 형식으로 저장"""
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # 헤더
            writer.writerow(['항목', '값'])

            # 기본 정보
            writer.writerow(['회사명', report.company_info.name])
            writer.writerow(['업종', report.company_info.industry])
            writer.writerow(['평가일', report.evaluation_date.strftime('%Y-%m-%d')])

            # 점수 정보
            writer.writerow(['총점', report.unicorn_score.total_score])
            writer.writerow(['등급', report.unicorn_score.grade])
            writer.writerow(['유니콘확률', f"{report.unicorn_score.unicorn_probability:.1%}"])
            writer.writerow(['투자추천', report.recommendation.value])

            # 영역별 점수
            writer.writerow(['', ''])  # 빈 줄
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
            "analysis_count": len(report.analysis_results),
            "risk_count": len(report.risk_assessments),
            "high_risks": len([r for r in report.risk_assessments
                             if r.risk_level.value in ["높음", "매우 높음"]]),
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
        output_path: str = None,
        pdf_output_path: str = None
    ) -> str:
        """최종 출력 처리"""

        # 실행 완료 시간 기록
        context.execution_end_time = datetime.now()

        if output_format == "console":
            output = self.output_processor.format_console_output(context)

            # 파일 저장 옵션
            if save_to_file and output_path and context.final_report:
                self.output_processor.save_report_to_file(
                    context.final_report, output_path, "json"
                )

            # PDF 저장 옵션
            if pdf_output_path and context.final_report:
                success = self.output_processor.save_report_to_file(
                    context.final_report, pdf_output_path, "pdf"
                )
                if success:
                    output += f"\n\n[완료] PDF 보고서 생성 완료: {pdf_output_path}\n"
                else:
                    output += f"\n\n[오류] PDF 보고서 생성 실패\n"

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

    def print_processing_summary(self, context: PipelineContext) -> None:
        """처리 과정 요약 출력"""
        print("\n" + "="*60)
        print("[처리] 파이프라인 실행 요약")
        print("="*60)

        if context.execution_start_time and context.execution_end_time:
            duration = context.execution_end_time - context.execution_start_time
            print(f"[시간] 총 실행 시간: {duration.total_seconds():.1f}초")

        print(f"[분석] 처리된 단계: {len(context.processing_steps)}개")

        for i, step in enumerate(context.processing_steps, 1):
            print(f"{i:2d}. {step}")

        if context.quality_check:
            print(f"\n 최종 품질 점수: {context.quality_check.overall_quality:.1%}")

        print("="*60)

def create_output_layer() -> OutputLayer:
    """Output Layer 생성자"""
    return OutputLayer()

def process_output_layer(
    context: PipelineContext,
    output_format: str = "console",
    save_to_file: bool = False,
    output_path: str = None,
    pdf_output_path: str = None
) -> str:
    """Output Layer 처리 함수"""
    output_layer = create_output_layer()

    # 최종 출력 생성
    final_output = output_layer.process_final_output(
        context=context,
        output_format=output_format,
        save_to_file=save_to_file,
        output_path=output_path,
        pdf_output_path=pdf_output_path
    )

    # 처리 과정 요약 (콘솔 모드에서만)
    if output_format == "console":
        output_layer.print_processing_summary(context)

    return final_output