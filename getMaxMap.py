import csv

file_path = "runs/train/MTDT5C/yolov6s-update2/results.csv"

map50 = []
map95 = []
P = []
R = []
map50_and95 = []

with open(file_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        map50.append(float(row["       metrics/mAP50(B)"]))
        map95.append(float(row["    metrics/mAP50-95(B)"]))
        P.append(float(row["   metrics/precision(B)"]))
        R.append(float(row["      metrics/recall(B)"]))
        map50_and95.append(float(row["       metrics/mAP50(B)"]) * 0.10 + float(row["    metrics/mAP50-95(B)"]) * 0.90)

# 选出map50和map50-95综合最大的一个数据，比例为1:9
max_map50 = max(map50)
max_map50_95 = max(map50_and95)
# if max_map50 > max_map50_95:
#     index = map50.index(max_map50)
# else:
index = map50_and95.index(max_map50_95)

print("epoch: ", index)
print("map50: ", map50[index])
print("map50-95: ", map95[index])
print("map_all: ", max_map50_95)
print("precision: ", P[index])
print("recall: ", R[index])
