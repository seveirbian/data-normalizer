from extrinsic_checker.report import Verdict, print_report


def test_print_report_all_pass(capsys):
    vs = [Verdict("head", "depth", True, {"plane_height": 0.84}, ["a.ply"]),
          Verdict("hand_left", "projection", True, {}, ["b.png"])]
    overall = print_report(vs)
    out = capsys.readouterr().out
    assert overall is True
    assert "head" in out and "PASS" in out


def test_print_report_one_fail(capsys):
    vs = [Verdict("head", "depth", True, {}, []),
          Verdict("hand_left", "projection", False, {}, [])]
    overall = print_report(vs)
    assert overall is False
    assert "FAIL" in capsys.readouterr().out
