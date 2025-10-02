"""
이모지 제거 스크립트 - cp949 인코딩 문제 해결
"""
import os
import re

# 이모지 매핑
emoji_map = {
    '🚀': '[시작]',
    '🔍': '[검색]',
    '📚': '[문서]',
    '💰': '[재무]',
    '📊': '[분석]',
    '✅': '[완료]',
    '❌': '[오류]',
    '⚠️': '[경고]',
    '🎯': '[목표]',
    '📈': '[성장]',
    '🔄': '[처리]',
    '📄': '[파일]',
    '🏆': '[결과]',
    '💡': '[정보]',
    '🌐': '[웹]',
    '📝': '[보고서]',
    '🤖': '[AI]',
    '📰': '[뉴스]',
    '🧪': '[테스트]',
    '🎉': '[성공]',
}

def remove_emojis_from_file(filepath):
    """파일에서 이모지 제거"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # 이모지 교체
        for emoji, replacement in emoji_map.items():
            content = content.replace(emoji, replacement)

        # 변경사항이 있으면 저장
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"수정: {filepath}")
            return True
        return False
    except Exception as e:
        print(f"오류 ({filepath}): {e}")
        return False

# layers 폴더의 모든 .py 파일 처리
layers_dir = 'D:/RAG_Project_test/layers'
count = 0

for filename in os.listdir(layers_dir):
    if filename.endswith('.py'):
        filepath = os.path.join(layers_dir, filename)
        if remove_emojis_from_file(filepath):
            count += 1

# pipeline.py도 처리
if remove_emojis_from_file('D:/RAG_Project_test/pipeline.py'):
    count += 1

print(f"\n총 {count}개 파일 수정 완료")
