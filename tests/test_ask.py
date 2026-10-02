from __future__ import annotations

import unittest

from ness_agent.tools.ask import question, set_question_runtime


class QuestionToolTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self) -> None:
        set_question_runtime(None)

    def test_schema_requires_questions(self) -> None:
        schema = question.args_schema.model_json_schema()
        self.assertIn("questions", schema.get("required", []))

    async def test_empty_questions_fails_schema_validation(self) -> None:
        with self.assertRaises(Exception):
            await question.ainvoke({"questions": []})

    async def test_valid_questions_without_handler(self) -> None:
        result = await question.ainvoke(
            {
                "questions": [
                    {
                        "prompt": "Which backend?",
                        "options": [
                            {"id": "redis", "label": "Redis", "recommended": True},
                            {"id": "memory", "label": "In-memory"},
                        ],
                    }
                ]
            }
        )
        self.assertIn("no interactive question handler", result)


if __name__ == "__main__":
    unittest.main()
