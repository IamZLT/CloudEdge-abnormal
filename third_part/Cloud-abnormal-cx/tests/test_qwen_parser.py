import ast
import pathlib
import unittest


def load_parser_functions():
    path = pathlib.Path(__file__).parents[1] / "cloud_abnormal" / "qwen.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = [
        node
        for node in tree.body
        if isinstance(node, ast.Import)
        and any(alias.name in {"json", "re", "warnings"} for alias in node.names)
    ]
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_recover_fields", "_extract_json"}
    ]
    namespace = {}
    module = ast.fix_missing_locations(ast.Module(body=imports + functions, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["_extract_json"]


class QwenParserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parse = staticmethod(load_parser_functions())

    def test_valid_json(self):
        result = self.parse(
            '{"anomaly_probability":0.8,"defect_type":"scratch",'
            '"regions":[],"reason":"small mark"}'
        )
        self.assertEqual(result["anomaly_probability"], 0.8)

    def test_missing_comma_is_repaired(self):
        result = self.parse(
            '{"anomaly_probability":0.8\n'
            '"defect_type":"scratch","regions":['
            '{"bbox":[1,2,3,4],"confidence":0.9,"defect":"scratch"}],'
            '"reason":"small mark"}'
        )
        self.assertEqual(result["anomaly_probability"], 0.8)
        self.assertEqual(result["regions"][0]["bbox"], [1, 2, 3, 4])
        self.assertTrue(result["parse_repaired"])

    def test_non_json_falls_back(self):
        result = self.parse("The image is probably normal.")
        self.assertEqual(result["anomaly_probability"], 0.0)
        self.assertEqual(result["regions"], [])
        self.assertTrue(result["parse_fallback"])


if __name__ == "__main__":
    unittest.main()
