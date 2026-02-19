# -*- coding: utf-8 -*-
"""
Simulator Tools for Student Agent Evaluation
=============================================

Three async tools for student-agent simulation, each workspace-isolated:

* :func:`solve_question` — solve a question with full memory pipeline
* :func:`generate_questions` — generate MC questions (correct answers hidden)
* :func:`submit_answers` — submit answers, auto-judge, trigger memory update
"""

from .tools import generate_questions, solve_question, submit_answers

__all__ = ["solve_question", "generate_questions", "submit_answers"]
