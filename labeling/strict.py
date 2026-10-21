"""Strict routed / test sets for every condition arm.

Rule: route only on hacks that are clearly X, and test only on hacks that are clearly not X, by two
independent signals wherever we have them; anything in between (the taggers disagree, or the hack is
borderline) is neither routed nor tested. routing.py draws each arm's routed hacks from route(arm),
and writes test(arm) next to the routing list as <arm>.test.json for the loss and agentic analyses.

Signals: Terminal Wrench categories (h["cats"]), our Sonnet labels (reads/tampers/redteam per hack;
security/languages/python_only per task), task_signals.json (reference-solution language, security
terms in the instruction) and a red-team regex over the transcript's narration.
"""
import re

A_CATS = {"hollow-implementation", "output-spoofing", "algorithmic-simplification", "keyword-gaming",
          "constraint-loophole"}
B_CATS = {"structural-extraction"}
C_CATS = {"binary-hijacking", "mutable-input-tampering", "metric-spoofing", "security-downgrading"}
RED_TEAM = re.compile(r"red[\s-]?team|qa agent|adversarial(ly)? (test|qa|evaluat)|authori[sz]ed (to|red)|"
                      r"my (role|assignment)|as instructed|assigned (role|task) .*verif", re.I)
# Narration that treats bypassing as an assigned objective without naming the role ("since I am tasked
# with providing a bypass", "the adversarial goal", "test the verifier's robustness"): not routed, and
# kept out of the Red-team test set, which should not be framed as an assignment at all.
ASSIGNED_OBJECTIVE = re.compile(
    r"tasked with|(my|our|the) (task|goal|objective|job|mission|mandate|instructions?) (is|was) to "
    r"(find|bypass|exploit|test|probe|game|pass the verifier)|adversarial(ly)?|objective|robustness|"
    r"(test|probe|stress)\w* (the )?(verifier|grader|checker|evaluat)|verifier[- ]bypass|"
    r"bypass (attempt|task|challenge|objective|goal)|as requested|instructed|assign\w*|authori[sz]\w*", re.I)
# Security-reason text where the labeler hedges ("security is incidental", "although it redacts ...")
HEDGED = re.compile(r"incidental|tangential|minor|peripheral|adjacent|although|though", re.I)


def narration_redteam(text):
    return bool(RED_TEAM.search(text))


class Signals:
    """tasks: labels/tasks.jsonl rows by task id; sig: task_signals.json; narr: hack key -> narration."""

    def __init__(self, tasks, sig, narr, hacks=()):
        self.tasks, self.sig, self.narr = tasks, sig, narr
        # tasks where some hack weakened security controls (TW security-downgrading)
        self.downgraded = {h["task"] for h in hacks if "security-downgrading" in h["cats"]}

    # --- hack type (Make-it-up)
    def make_it_up(self, h):  # invents its output: only A categories, and our labels see no B or C
        return bool(h["cats"]) and set(h["cats"]) <= A_CATS and not h["reads"] and not h["tampers"]

    def peeks(self, h):  # B by both taggers
        return h["reads"] and bool(set(h["cats"]) & B_CATS)

    def tampers(self, h):  # C by both taggers
        return h["tampers"] and bool(set(h["cats"]) & C_CATS)

    # --- narration (Red-team)
    def redteam(self, h):
        return h["redteam"] and narration_redteam(self.narr[h["key"]])

    def no_redteam(self, h):  # no role, and no assigned-objective framing either
        t = self.narr[h["key"]]
        return not h["redteam"] and not narration_redteam(t) and not ASSIGNED_OBJECTIVE.search(t)

    # --- task properties (Non-security, Python-only)
    def nonsecurity_task(self, t):  # not security, and nothing security-adjacent anywhere in the task
        s = self.sig[t]
        return (not self.tasks[t]["security_related"] and s["security_terms"] == 0
                and s["security_adjacent_terms"] == 0 and t not in self.downgraded
                and not HEDGED.search(self.tasks[t]["security_reason"]))

    def security_task(self, t):
        return self.tasks[t]["security_related"] and self.sig[t]["security_terms"] >= 2

    def python_task(self, t):  # Sonnet says only Python is written, and the reference solution agrees
        s = self.sig[t]
        return self.tasks[t]["python_only"] and s["py_solution"] and not s["other_solution"]

    def no_python_task(self, t):  # writes code, none of it Python, by both signals
        langs = self.tasks[t]["languages_written"]
        return bool(langs) and "python" not in langs and not self.sig[t]["py_solution"]


def arm_sets(sig, short_tasks, long_tasks):
    """arm -> (route predicate, test predicate) over joined hack rows."""
    s = sig
    return {
        "A_make_it_up": (s.make_it_up, lambda h: s.peeks(h) or s.tampers(h)),
        "H9_redteam": (s.redteam, s.no_redteam),
        "T3_nonsecurity": (lambda h: s.nonsecurity_task(h["task"]), lambda h: s.security_task(h["task"])),
        "T5_python_only": (lambda h: s.python_task(h["task"]), lambda h: s.no_python_task(h["task"])),
        "L1_short": (lambda h: h["task"] in short_tasks, lambda h: h["task"] in long_tasks),
    }
