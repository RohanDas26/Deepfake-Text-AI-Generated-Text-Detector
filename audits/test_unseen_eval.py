"""
Evaluation of the new pipeline on 12 completely unseen samples using run_inference (no overrides).
"""
import os, sys, logging
logging.basicConfig(level=logging.WARNING)

from tasks import run_inference, LABELS

UNSEEN = [
    # Class 1: FULLY AI-GENERATED
    {"label": 1, "source": "GPT-4 article intro",
     "text": "The rapid proliferation of large language models has fundamentally altered the technological landscape, enabling unprecedented automation of complex cognitive tasks. These systems demonstrate remarkable proficiency across diverse domains, from scientific summarization to creative content generation. As adoption accelerates, researchers are increasingly focused on alignment, interpretability, and robustness."},
    {"label": 1, "source": "Gemini essay paragraph",
     "text": "Sustainable urban development requires the integration of green infrastructure within city planning frameworks. By incorporating parks, green roofs, and permeable pavements, municipalities can effectively manage stormwater runoff while reducing urban heat island effects. These interventions collectively improve resident quality of life and ecological resilience."},
    {"label": 1, "source": "Claude formal writing",
     "text": "The epistemological foundations of empirical science rest upon the principle of falsifiability, as articulated by Karl Popper. A claim possesses scientific validity only if it is theoretically possible to demonstrate its incorrectness through observation. This criterion distinguishes scientific inquiry from metaphysical speculation."},
    {"label": 1, "source": "ChatGPT casual reply",
     "text": "Of course! Here are some effective study strategies for your upcoming exam. First, use active recall techniques by testing yourself regularly rather than passively rereading notes. Second, space out your study sessions over multiple days using the spaced repetition method. Finally, prioritize understanding core concepts over memorizing specific details."},

    # Class 0: HUMAN-WRITTEN
    {"label": 0, "source": "Reddit casual post",
     "text": "honestly my laptop fan has been going insane for the last week and i have no idea why. tried closing chrome tabs but it still sounds like a jet engine. my roommate says just buy a new one but like, i'm a broke student, that's not happening lol"},
    {"label": 0, "source": "Personal blog",
     "text": "Three weeks into learning Portuguese and I can confidently say I now know how to order a coffee and apologize for things that aren't my fault. The subjunctive tense is going to be the death of me though. Seriously, why does every sentence need seventeen different verb forms."},
    {"label": 0, "source": "Stack Overflow rant",
     "text": "I've been wrestling with this CORS issue for two days now. The preflight request keeps returning 403 despite setting Access-Control-Allow-Origin to *. I've checked nginx config, FastAPI middleware, and I'm about ready to give up and just disable the browser security entirely for testing."},
    {"label": 0, "source": "WhatsApp message style",
     "text": "bro did you see that match yesterday?? insane ending honestly didn't expect them to come back from 3-0 down like that. my whole family was freaking out haha"},

    # Class 2: HUMAN-POLISHED / AI-PARAPHRASED
    {"label": 2, "source": "QuillBot paraphrase of GPT-4",
     "text": "The swift expansion of AI-driven language systems has dramatically reshaped the digital landscape by enabling unprecedented automation of intricate mental tasks. These tools show impressive capability spanning multiple fields, ranging from scientific summarization to creative content production."},
    {"label": 2, "source": "Student AI-assisted essay",
     "text": "Climate change is honestly one of the scariest issues of our generation. Rising global temperatures are contributing to extreme weather events and higher sea levels, which threatens millions of coastal communities. International agreements like the Paris Accord are important, but individual countries need to actually follow through on their pledges."},
    {"label": 2, "source": "Grammarly-polished email",
     "text": "I wanted to reach out to discuss the Q3 project timeline. While we have made meaningful progress across several workstreams, a few deliverables are at risk of slipping due to resource constraints. I think a short sync this week could help us realign priorities and keep the project on track."},
    {"label": 2, "source": "ChatGPT lightly edited by human",
     "text": "You know, sometimes I feel like one of the most interesting phases of life is the one where you know you want something big, but haven't quite figured out how to get there yet. You're building skills, trying new things, making plans. Here's something worth keeping in mind: you don't necessarily need to have your future figured out right now."},
]

label_names = {0: "HUMAN-WRITTEN", 1: "FULLY AI-GENERATED", 2: "HUMAN-POLISHED / AI-PARAPHRASED"}

print("\n" + "="*70)
print("EVALUATING RECALIBRATED PIPELINE ON 12 UNSEEN SAMPLES")
print("="*70)

y_true = []
y_pred = []

for sample in UNSEEN:
    result = run_inference(sample["text"], compute_attributions=False)
    pred_label = result["label"]
    conf = result["confidence"]
    metrics = result["metrics"]
    
    # Map back to int
    pred_idx = [k for k, v in LABELS.items() if v == pred_label][0]
    true_idx = sample["label"]
    y_true.append(true_idx)
    y_pred.append(pred_idx)
    
    status = "CORRECT" if pred_idx == true_idx else "WRONG"
    print(f"[{status:7s}] Source: {sample['source']}")
    print(f"          True: {label_names[true_idx]}")
    print(f"          Pred: {pred_label} (Confidence: {conf:.1%})")
    print(f"          PPL: {metrics['perplexity']:.1f} | Burstiness: {metrics['burstiness']:.3f} | TTR: {metrics['ttr']:.3f}")
    print(f"          DeBERTa: {metrics['deberta_label']} ({metrics['deberta_conf']:.1%}) | RF: {metrics['rf_label']} ({metrics['rf_conf']:.1%})")
    print()

from sklearn.metrics import classification_report, confusion_matrix
print("="*70)
print("UNSEEN EVALUATION CLASSIFICATION REPORT:")
print("="*70)
print(classification_report(y_true, y_pred, target_names=["HUMAN", "AI", "PARAPHRASED"], zero_division=0))

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix (rows=true, cols=pred):")
print(f"{'':15s} {'HUMAN':>12s} {'AI':>12s} {'PARAPH':>12s}")
names = ["HUMAN", "AI", "PARAPHRASED"]
for i, row in enumerate(cm):
    print(f"  {names[i]:13s}: {row[0]:>12d} {row[1]:>12d} {row[2]:>12d}")
