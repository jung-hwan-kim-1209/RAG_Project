"""
모든 이모지 제거 (정규식 사용)
"""
import os
import re

def remove_all_emojis(text):
    """모든 이모지 제거 (정규식)"""
    # 이모지 범위 패턴
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002700-\U000027BF"  # Dingbats
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002600-\U000026FF"  # Miscellaneous Symbols
        u"\U0001F700-\U0001F77F"  # Alchemical Symbols
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def process_file(filepath):
    """파일에서 이모지 제거"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        new_content = remove_all_emojis(content)

        if content != new_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"수정: {filepath}")
            return True
        return False
    except Exception as e:
        print(f"오류 ({filepath}): {e}")
        return False

# 모든 Python 파일 처리
count = 0

# layers 폴더
layers_dir = 'D:/RAG_Project_test/layers'
for filename in os.listdir(layers_dir):
    if filename.endswith('.py'):
        filepath = os.path.join(layers_dir, filename)
        if process_file(filepath):
            count += 1

# 루트 파일들
for filename in ['cli.py', 'pipeline.py', 'models.py']:
    filepath = f'D:/RAG_Project_test/{filename}'
    if os.path.exists(filepath) and process_file(filepath):
        count += 1

print(f"\n총 {count}개 파일 수정 완료")
