from metrics import Grader


def test_grader():
    grader = Grader()
    preds = ["Hello, world!", "Hello, world!", "Hello, world!"]
    refs = ["Hello, world!", "Hello, world!", "Hello, world!"]
    random_string_to_prepends = ["", "", ""]
    grades, error_details = grader.grade(preds, refs, random_string_to_prepends)
    print(grades)
    print(error_details)
    assert grades == [1.0, 1.0, 1.0]
    assert error_details == [None, None, None]


if __name__ == "__main__":
    test_grader()
