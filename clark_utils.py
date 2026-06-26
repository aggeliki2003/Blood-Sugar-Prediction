import numpy as np

def calculate_clarke_zones(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    zones = {"A":0, "B":0, "C":0, "D":0, "E":0}

    for r, p in zip(y_true, y_pred):

        # Zone A
        if (r <= 70 and p <= 70) or (abs(r - p) <= 0.2 * r):
            zones["A"] += 1

        # Zone E (confusing hypo/hyper treatment)
        elif (r <= 70 and p >= 180) or (r >= 180 and p <= 70):
            zones["E"] += 1

        # Zone D (dangerous failure to detect hypo/hyper)
        elif (r <= 70 and 70 < p < 180) or (r >= 240 and 70 <= p <= 180):
            zones["D"] += 1

        # Zone C (overcorrecting acceptable values)
        elif (70 <= r <= 290 and p >= r + 110) or (130 <= r <= 180 and p <= (7/5)*(r-130)):
            zones["C"] += 1

        # Zone B (benign errors)
        else:
            zones["B"] += 1

    total = len(y_true)

    percentages = {k: (v/total)*100 for k,v in zones.items()}
    percentages["A+B"] = ((zones["A"] + zones["B"]) / total) * 100

    return percentages