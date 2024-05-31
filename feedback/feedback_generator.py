# generate feedback JSON file
import numpy as np
import json

class feedback(object):

    def __init__(self, output_path):
        self.output_path = output_path

    # score feedback of each aspect
    def ABS_SHAP(self, shap):
        df = shap.T * 100
        df = df.reset_index()
        df.columns = ['Variable', 'SHAP']
        df['Sign'] = np.where(df['SHAP'] > 0, 'red', 'blue')
        df['SHAP'] = df[['SHAP']].apply(lambda x: np.abs(x))
        df = df.sort_values(by='SHAP', ascending=True)
        colorlist = df['Sign']
        ax = df.plot.barh(x='Variable', y='SHAP', color=colorlist, figsize=(25, 15), legend=False, fontsize=25)
        ax.set_ylabel("")
        ax.set_xlabel(
            "Contribution to the final score of each behaviour. Blue means negative Red means positive",
            fontsize=30)
        ax.figure.savefig('{}/feedback.png'.format(self.output_path))
        return df

    # polish feature name
    def polish(self, name):
        if "_" in name:
            name = " ".join(name.split('_'))
        return name

    # map SHAP value to 5 likert scale
    def scoreMap(self, num):
        if num > 1:
            return 5
        elif 0.2 < num <= 1:
            return 4
        elif -0.2 <= num <= 0.2:
            return 3
        elif -1 < num < -0.2:
            return 2
        else:
            return 1

    # generate report content
    def generateReport(self, group_shap, score, key):
        report = {"confidenceScore": int(score * 100)}
        fe = round(group_shap.iloc[0]["facial_expression"] / 0.234, 3)
        fluency = round(group_shap.iloc[0]["fluency"] / 0.027, 3)
        pros = round(group_shap.iloc[0]["prosody"] / 0.113, 3)
        vq = round(group_shap.iloc[0]["voice_quality"] / 0.06, 3)
        fe = self.scoreMap(fe)
        fluency = self.scoreMap(fluency)
        pros = self.scoreMap(pros)
        vq = self.scoreMap(vq)

        df = self.ABS_SHAP(group_shap)
        df = df.sort_values(by='SHAP', ascending=False)
        summary = ""
        fluency_r = ""
        fe_r = ""
        pros_r = ""
        vq_r = ""
        if 0.7 >= score > 0.5:
            summary += "Not so bad, at least you act confident!"
        elif score > 0.7:
            summary += "Good job, you perform really confident!"
        else:
            summary += "Here are the points of improvement."
        count = 0
        for _, row in df.iterrows():
            if row["Sign"] == "blue" and count == 0:
                summary += " We suggest that you should first improve {}.".format(self.polish(row["Variable"]))
                count += 1
            elif row["Sign"] == "blue" and count == 1:
                summary += " Second, you should focus on improving {}.".format(self.polish(row["Variable"]))
                break

        if fluency > 3:
            fluency_r += "You speak fluently."
        elif fluency < 3:
            fluency_r += "Your speech is not fluent."
        else:
            fluency_r += "Your speech fluency is average."
        if pros >3:
            pros_r += "Your prosody is comfortable."
        elif pros<3:
            pros_r += "You should improve your prosody."
        else:
            pros_r += "Your speech prosody is average."
        if fe > 3:
            fe_r += "Your facial expression is good, keep it!"
        elif fe==3:
            fe_r += "Your facial expression is average."
        else:
            fe_r += "Your facial expression is bad, please improve it, it's very important!"
        if vq > 3:
            vq_r += "Your voice quality is good."
        elif vq==3:
            vq_r += "Your voice quality is average."
        else:
            vq_r += "You need to improve your voice quality."

        report["audio"] = {"prosody": {"pros_tag": True if pros > 3 else False, "pros_r": pros_r},
                           "vq": {"vq_tag": True if vq > 3 else False, "vq_r": vq_r}, "audio_tag": True,
                           "pros_s": pros, "vq_s": vq}
        report["text"] = {"fluency": {"fluency_tag": True if fluency > 3 else False, "fluency_r": fluency_r},
                          "fluency_s": fluency}
        report["visual"] = {"fe": {"fe_r": fe_r}, "fe_s": fe}
        report["key"] = key
        report["summary"] = summary
        with open('{}/report.json'.format(self.output_path), "w") as f:
            json.dump(report, f)