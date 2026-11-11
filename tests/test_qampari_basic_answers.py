import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "src/evaluation/qampari/answer/basic_ans_evaluation.py"
SPEC = importlib.util.spec_from_file_location("qampari_basic", SOURCE)
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)
GOLD = [{"answer_text": name, "aliases": [name, f"{name} alias"]}
        for name in ("Alpha", "Beta", "Gamma")]


def record(generation):
    return {"ex_num": 1, "query": "Synthetic question", "generation": generation,
            "info": {"answer_list": GOLD}}


@pytest.mark.parametrize(("generation", "expected"), [
    ("Answers: Alpha, Beta, Gamma", ["Alpha", "Beta", "Gamma"]),
    ("Alpha\nBeta\nGamma", ["Alpha", "Beta", "Gamma"]),
    ("Alpha#Beta#Gamma", ["Alpha", "Beta", "Gamma"]),
    ("Answers: Alpha, Beta\nGamma", ["Alpha", "Beta", "Gamma"]),
    ("Alpha\nBeta, Gamma", ["Alpha", "Beta", "Gamma"]),
    ("Alpha,#\nBeta\n#Gamma,", ["Alpha", "Beta", "Gamma"]),
    ("Alpha,,,Beta,", ["Alpha", "Beta"]),
    ("Alpha\n\nBeta\n", ["Alpha", "Beta"]),
    ("#Alpha##Beta#", ["Alpha", "Beta"]),
    ("", []), ("  \n\t ", []), (None, []), ("Answers: ", []),
    ("Alpha", ["Alpha"]), ("Answer: Alpha", ["Alpha"]),
    (" aNsWeRs: Alpha ", ["Alpha"]), (" aNsWeR: Alpha ", ["Alpha"]),
    (",", [","]), ("#", ["#"]),
    ("Washington, D.C.", ["Washington", "D.C."]),
    ("C#", ["C"]),
])
def test_supported_formats(generation, expected):
    assert EVALUATOR.parse_generation_to_predictions(generation) == expected


@pytest.mark.parametrize("generation", ["Answers: Alpha, Beta\nGamma", "Alpha,Beta,Gamma",
                                        "Alpha\nBeta\nGamma", "Alpha#Beta#Gamma",
                                        "THE ALPHA ALIAS, Beta alias#Gamma alias"])
def test_transformation_metrics_and_details(generation):
    transformed = EVALUATOR.transform_llm_results_format([record(generation)])
    assert len(transformed[0]["predictions"]) == 3
    assert transformed[0]["answer_list"] == GOLD
    assert transformed[0]["query"] == "Synthetic question"
    metrics = EVALUATOR.compute_metrics_qampari(transformed)
    assert metrics == {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                       "f1_above": 1.0, "rec_above": 1.0}
    aggregate, details = EVALUATOR.compute_metrics_qampari_with_details(transformed)
    assert aggregate["avg_f1"] == 1.0
    assert details[0]["num_correct"] == 3
    assert details[0]["predicted_answers_num"] == 3


def test_validated_answer_input_and_preparsed_input():
    item = {"generation": "Alpha\nBeta, Gamma", "validated_answer": GOLD}
    transformed, = EVALUATOR.transform_llm_results_format([item])
    assert transformed["predictions"] == ["Alpha", "Beta", "Gamma"]
    existing = {"answer_list": GOLD, "predictions": ["Alpha", "wrong"]}
    transformed = EVALUATOR.transform_llm_results_format([existing])
    metrics = EVALUATOR.compute_metrics_qampari(transformed)
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == pytest.approx(1 / 3)
    assert metrics["f1"] == 0.4


def test_cli_detailed_evaluation(tmp_path):
    source = tmp_path / "input.json"
    output = tmp_path / "metrics.json"
    source.write_text(json.dumps([record("Answers: Alpha, Beta\nGamma")]), encoding="utf-8")
    run = subprocess.run([sys.executable, str(SOURCE), "--input-path", str(source),
                          "--save-path", str(output)], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["avg_precision"] == payload["avg_recall"] == payload["avg_f1"] == 1.0
    assert set(payload["results"][0]["parsed_predictions"]) == {"Alpha", "Beta", "Gamma"}
