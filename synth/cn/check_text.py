from collections import Counter
import matplotlib.pyplot as plt
import string

# 檔案路徑
txt_path = "/media/avlab/Transcend/Synthplate/Synthtext/data/plate_list_final2.txt"

char_counter = Counter()
valid_samples = 0

with open(txt_path, "r", encoding="utf-8") as f:
    for line in f:
        plate = line.strip()

        if not plate:
            continue

        # 移除 '-'
        plate_no_dash = plate.replace("-", "")

        # 只統計長度為 6 或 7 的車牌
        if len(plate_no_dash) not in [6, 7]:
            continue

        valid_samples += 1

        # 統計字元
        for ch in plate_no_dash:
            if ch.isalnum():
                char_counter[ch.upper()] += 1

print(f"Valid samples: {valid_samples}")
print(f"Total characters: {sum(char_counter.values())}")

print("\nCharacter Frequency:")
for char, count in sorted(char_counter.items()):
    print(f"{char}: {count}")

# 為了讓圖表順序固定：0-9 + A-Z
ordered_chars = list(string.digits) + list(string.ascii_uppercase)
total_chars = sum(char_counter.values())
freqs_percent = [
    char_counter.get(ch, 0) / total_chars * 100
    for ch in ordered_chars
]

plt.figure(figsize=(16, 6))
plt.bar(ordered_chars, freqs_percent)
plt.xlabel("Character")
plt.ylabel("Percentage (%)")
plt.title("License Plate Character Distribution")
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()