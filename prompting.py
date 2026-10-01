"""Prompt rendering: turn a scenario + history into the exact string the model sees.

This is the same substitution used in the experiments (see pipeline/run_all.py),
extracted here without the torch dependency so the demo/ scripts run on a laptop.

A prompt has five parts:
    1. scenario header with the round index
    2. hidden parameter space + observation model  (static per scenario)
    3. the rendered history of completed rounds    {history}
    4. the current round and its two labelled options
    5. a task instruction

Parts 1-4 are the "scenario body" and are identical for every probe. Only the
task instruction differs:

    Probes 1 and 2 (forced choice)  prompts/{env}_forced.txt
        ends "reply with the single letter of the best action. Do not explain."
        No tokens are generated; the model is read from one forward pass.

    Probe 3 (chain of thought)      prompts/{env}_cot.txt
        ends with a numbered reasoning instruction whose final step is
        {answer_format}.

LABEL SWAPPING. Every prompt is rendered under both assignments of semantic
actions to answer letters. Under mapping 0, the scenario's action 0 is shown as
"A"; under mapping 1 the two are exchanged. Averaging the A/B logit difference
over the two mappings cancels any intrinsic preference the model has for a
particular letter, which is otherwise large enough to dominate the measurement
in small models.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT_DIR = os.path.join(HERE, "prompts")

# Default answer-format instruction, used by the seven instruction-tuned models.
ANSWER_FORMAT_LETTER = (
    "Write your final answer \u2014 exactly one of:\n"
    "     Answer: A\n"
    "     Answer: B"
)

# The two R1-Distill reasoning models instead receive the phrasing recommended
# by their own model cards. This is a format-compliance instruction only: the
# scenario, evidence, options and reasoning steps are unchanged.
ANSWER_FORMAT_BOXED = (
    "Please reason step by step, and put your final answer within \\boxed{}, "
    "exactly one of:\n"
    "     \\boxed{A}\n"
    "     \\boxed{B}"
)


def load_template(env_name, kind):
    """kind is 'forced' (Probes 1-2) or 'cot' (Probe 3)."""
    with open(os.path.join(PROMPT_DIR, f"{env_name}_{kind}.txt")) as fh:
        return fh.read()


def render(template, env, t, T, history, ctx, mapping,
           answer_format=ANSWER_FORMAT_LETTER):
    """mapping 0: A = action 0, B = action 1 (canonical). mapping 1: swapped."""
    if mapping == 0:
        a_desc, b_desc = env.act0_desc, env.act1_desc
    else:
        a_desc, b_desc = env.act1_desc, env.act0_desc
    fields = {
        "t": t,
        "T": T,
        "history": history,
        "option_A_desc": a_desc,
        "option_B_desc": b_desc,
        "answer_format": answer_format,
        **ctx,
    }
    return template.format(**fields)


def rollout(env, T, upto=None):
    """Run the scripted behaviour policy and return (records, ctx_by_round).

    The evidence stream is produced by a fixed behaviour policy, NOT by the
    evaluated model, so every model sees an identical history for a given seed.
    """
    records, ctxs = [], {}
    for t in range(1, (upto or T) + 1):
        ctx = env.new_round(t)
        ctxs[t] = ctx
        a = env.behavior_action(t)
        _, rec = env.outcome(t, ctx, a)
        records.append(rec)
    return records, ctxs
