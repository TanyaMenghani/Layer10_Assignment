import google.generativeai as genai

genai.configure(api_key="AIzaSyCXho2RmhMl4c_M2WHmny-LPepBxFtIgFQ")

for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)