import json
import re
import pytest
from pathlib import Path

from ruleforge.optimize.input_output import parse_scenarios_to_jsonl as parse_with_pattern


SCENARIO_PATTERN = re.compile(
    r"^\s*# Scenario \d+\s*\n+"
    r"\s*Subset of Rule IDs:\s*(?P<subset_rule_ids>.*?)\s*\n+"
    r"\s*Possible Ambiguous Topics and Value Tradeoffs:\s*"
    r"(?P<ambiguous_topics_and_tradeoffs>.*?)\s*\n+"
    r"\s*Prompt:\s*(?P<prompt>.*?)\s*\n+"
    r"\s*Response:\s*(?P<response>.*?)(?=\n\s*# Scenario|\s*\Z)",
    re.DOTALL | re.MULTILINE,
)


def parse_scenarios_to_jsonl(input_str: str, output_path: str | Path) -> None:
    """Exercise the task-supplied pattern contract used by the trainer."""
    parse_with_pattern(SCENARIO_PATTERN, input_str, Path(output_path))


@pytest.fixture
def temp_jsonl_file(tmp_path):
    return tmp_path / "test_scenarios.jsonl"


def read_jsonl(file_path: Path):
    with file_path.open("r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def test_parse_single_scenario(temp_jsonl_file):
    input_str = """# Scenario 1
Subset of Rule IDs: Rule 1, Rule 3
Possible Ambiguous Topics and Value Tradeoffs: Privacy vs transparency
Prompt: What should I do about user data?
Response: You should balance privacy and transparency.
"""
    parse_scenarios_to_jsonl(input_str, str(temp_jsonl_file))

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 1
    assert result[0]["subset_rule_ids"] == "Rule 1, Rule 3"
    assert result[0]["ambiguous_topics_and_tradeoffs"] == "Privacy vs transparency"
    assert result[0]["prompt"] == "What should I do about user data?"
    assert result[0]["response"] == "You should balance privacy and transparency."


def test_parse_multiple_scenarios(temp_jsonl_file):
    input_str = """# Scenario 1
Subset of Rule IDs: Rule 1, Rule 2
Possible Ambiguous Topics and Value Tradeoffs: Safety vs convenience
Prompt: How to handle this situation?
Response: Prioritize safety.
# Scenario 2
Subset of Rule IDs: Rule 3, Rule 5
Possible Ambiguous Topics and Value Tradeoffs: Speed vs accuracy
Prompt: Which is more important?
Response: It depends on context.
# Scenario 3
Subset of Rule IDs: Rule 7
Possible Ambiguous Topics and Value Tradeoffs: Cost vs quality
Prompt: What's the best approach?
Response: Balance both factors.
"""
    parse_scenarios_to_jsonl(input_str, str(temp_jsonl_file))

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 3

    assert result[0]["subset_rule_ids"] == "Rule 1, Rule 2"
    assert result[0]["ambiguous_topics_and_tradeoffs"] == "Safety vs convenience"
    assert result[0]["prompt"] == "How to handle this situation?"
    assert result[0]["response"] == "Prioritize safety."

    assert result[1]["subset_rule_ids"] == "Rule 3, Rule 5"
    assert result[1]["ambiguous_topics_and_tradeoffs"] == "Speed vs accuracy"
    assert result[1]["prompt"] == "Which is more important?"
    assert result[1]["response"] == "It depends on context."

    assert result[2]["subset_rule_ids"] == "Rule 7"
    assert result[2]["ambiguous_topics_and_tradeoffs"] == "Cost vs quality"
    assert result[2]["prompt"] == "What's the best approach?"
    assert result[2]["response"] == "Balance both factors."


def test_parse_scenario_with_multiline_prompt(temp_jsonl_file):
    input_str = """# Scenario 1
Subset of Rule IDs: Rule 1, Rule 2
Possible Ambiguous Topics and Value Tradeoffs: Complex ethical dilemma
Prompt: This is a prompt
that spans multiple lines
and has various details.
Response: This is a single line response.
"""
    parse_scenarios_to_jsonl(input_str, str(temp_jsonl_file))

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 1
    assert result[0]["subset_rule_ids"] == "Rule 1, Rule 2"
    assert result[0]["ambiguous_topics_and_tradeoffs"] == "Complex ethical dilemma"
    assert "multiple lines" in result[0]["prompt"]
    assert result[0]["response"] == "This is a single line response."


def test_parse_scenario_with_multiline_response(temp_jsonl_file):
    input_str = """# Scenario 1
Subset of Rule IDs: Rule 4
Possible Ambiguous Topics and Value Tradeoffs: Detailed guidance needed
Prompt: Simple question?
Response: Here is a detailed response
that spans multiple lines.
It provides comprehensive guidance.
"""
    parse_scenarios_to_jsonl(input_str, str(temp_jsonl_file))

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 1
    assert result[0]["subset_rule_ids"] == "Rule 4"
    assert result[0]["ambiguous_topics_and_tradeoffs"] == "Detailed guidance needed"
    assert result[0]["prompt"] == "Simple question?"
    assert "multiple lines" in result[0]["response"]
    assert "comprehensive guidance" in result[0]["response"]


def test_parse_scenario_with_whitespace(temp_jsonl_file):
    input_str = """# Scenario 1
Subset of Rule IDs:   Rule 1, Rule 2, Rule 3   
Possible Ambiguous Topics and Value Tradeoffs:   Topic with spaces   
Prompt:   Prompt with leading/trailing spaces   
Response:   Response with spaces   
"""
    parse_scenarios_to_jsonl(input_str, str(temp_jsonl_file))

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 1
    assert result[0]["subset_rule_ids"] == "Rule 1, Rule 2, Rule 3"
    assert result[0]["ambiguous_topics_and_tradeoffs"] == "Topic with spaces"
    assert result[0]["prompt"] == "Prompt with leading/trailing spaces"
    assert result[0]["response"] == "Response with spaces"


def test_parse_scenario_with_empty_fields(temp_jsonl_file):
    input_str = """# Scenario 1
Subset of Rule IDs: 
Possible Ambiguous Topics and Value Tradeoffs: 
Prompt: What should I do?
Response: Do the right thing.
"""
    parse_scenarios_to_jsonl(input_str, str(temp_jsonl_file))

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 1
    assert result[0]["subset_rule_ids"] == ""
    assert result[0]["ambiguous_topics_and_tradeoffs"] == ""
    assert result[0]["prompt"] == "What should I do?"
    assert result[0]["response"] == "Do the right thing."


def test_parse_scenario_with_special_characters(temp_jsonl_file):
    input_str = """# Scenario 1
Subset of Rule IDs: Rule 1, Meta-Rule 2
Possible Ambiguous Topics and Value Tradeoffs: User privacy & data security → ethical considerations
Prompt: How to handle user's @email and password?
Response: Use encryption (AES-256) & secure hashing! 🔒
"""
    parse_scenarios_to_jsonl(input_str, str(temp_jsonl_file))

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 1
    assert result[0]["subset_rule_ids"] == "Rule 1, Meta-Rule 2"
    assert "→" in result[0]["ambiguous_topics_and_tradeoffs"]
    assert "@email" in result[0]["prompt"]
    assert "AES-256" in result[0]["response"]
    assert "🔒" in result[0]["response"]


def test_parse_scenario_with_colon_in_content(temp_jsonl_file):
    input_str = """# Scenario 1
Subset of Rule IDs: Rule 1:A, Rule 2:B
Possible Ambiguous Topics and Value Tradeoffs: Time format: HH:MM:SS
Prompt: What time is it: now or later?
Response: The time is 12:30:45 PM.
"""
    parse_scenarios_to_jsonl(input_str, str(temp_jsonl_file))

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 1
    assert ":" in result[0]["subset_rule_ids"]
    assert "HH:MM:SS" in result[0]["ambiguous_topics_and_tradeoffs"]
    assert "now or later?" in result[0]["prompt"]
    assert "12:30:45" in result[0]["response"]


def test_parse_scenario_with_numbered_list_in_rule_ids(temp_jsonl_file):
    input_str = """# Scenario 1
Subset of Rule IDs: [1, 3, 5, 7]
Possible Ambiguous Topics and Value Tradeoffs: Multiple conflicting rules
Prompt: Which rules apply here?
Response: All of them, in priority order.
"""
    parse_scenarios_to_jsonl(input_str, str(temp_jsonl_file))

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 1
    assert result[0]["subset_rule_ids"] == "[1, 3, 5, 7]"
    assert result[0]["ambiguous_topics_and_tradeoffs"] == "Multiple conflicting rules"


def test_parse_scenarios_with_varying_spacing(temp_jsonl_file):
    input_str = """# Scenario 1
Subset of Rule IDs: Rule 1
Possible Ambiguous Topics and Value Tradeoffs: Topic A
Prompt: Question 1?
Response: Answer 1.


# Scenario 2
Subset of Rule IDs: Rule 2
Possible Ambiguous Topics and Value Tradeoffs: Topic B
Prompt: Question 2?
Response: Answer 2.

# Scenario 3
Subset of Rule IDs: Rule 3
Possible Ambiguous Topics and Value Tradeoffs: Topic C
Prompt: Question 3?
Response: Answer 3.
"""
    parse_scenarios_to_jsonl(input_str, str(temp_jsonl_file))

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 3
    assert all("subset_rule_ids" in r for r in result)
    assert all("ambiguous_topics_and_tradeoffs" in r for r in result)
    assert all("prompt" in r for r in result)
    assert all("response" in r for r in result)


def test_parse_scenario_with_complex_multiline_content(temp_jsonl_file):
    input_str = """# Scenario 1
Subset of Rule IDs: Rule 1, Rule 2, Rule 3,
Rule 4, Rule 5
Possible Ambiguous Topics and Value Tradeoffs: This is a complex topic that involves:
- Privacy concerns
- Security requirements
- User experience
Prompt: User: Hi, I need help with something.
System: Sure, what do you need?
User: I want to share my data.
Response: System: I understand your request.
Let me walk you through the options:
1. Option A - Full sharing
2. Option B - Partial sharing
3. Option C - No sharing
"""
    parse_scenarios_to_jsonl(input_str, str(temp_jsonl_file))

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 1
    assert "Rule 5" in result[0]["subset_rule_ids"]
    assert "Privacy concerns" in result[0]["ambiguous_topics_and_tradeoffs"]
    assert "User: Hi" in result[0]["prompt"]
    assert "Option C" in result[0]["response"]


def test_parse_no_scenarios(temp_jsonl_file):
    input_str = """This is some random text
that doesn't match the scenario format.
It has multiple lines but no proper structure.
"""
    parse_scenarios_to_jsonl(input_str, str(temp_jsonl_file))

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 0


def test_parse_incomplete_scenario(temp_jsonl_file):
    input_str = """# Scenario 1
Subset of Rule IDs: Rule 1
Possible Ambiguous Topics and Value Tradeoffs: Some topic
"""
    parse_scenarios_to_jsonl(input_str, str(temp_jsonl_file))

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 0


def test_parse_scenario_with_json_in_content(temp_jsonl_file):
    input_str = """# Scenario 1
Subset of Rule IDs: Rule 1
Possible Ambiguous Topics and Value Tradeoffs: Data structures
Prompt: What format should I use?
Response: Use this format: {"key": "value", "count": 42}
"""
    parse_scenarios_to_jsonl(input_str, str(temp_jsonl_file))

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 1
    assert '{"key": "value"' in result[0]["response"]


def test_parse_scenario_with_code_blocks(temp_jsonl_file):
    input_str = """# Scenario 1
Subset of Rule IDs: Rule 7, Rule 8
Possible Ambiguous Topics and Value Tradeoffs: Code examples and documentation
Prompt: How do I implement this function?
Response: Here's an example:
```python
def example():
    return "Hello, World!"
```
"""
    parse_scenarios_to_jsonl(input_str, str(temp_jsonl_file))

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 1
    assert "```python" in result[0]["response"]
    assert "def example()" in result[0]["response"]


def test_file_creation_in_subdirectory(tmp_path):
    nested_path = tmp_path / "subdir1" / "subdir2" / "scenarios.jsonl"

    input_str = """# Scenario 1
Subset of Rule IDs: Rule 1
Possible Ambiguous Topics and Value Tradeoffs: Nested directories
Prompt: Does this work?
Response: Yes, it should.
"""
    parse_scenarios_to_jsonl(input_str, str(nested_path))

    assert nested_path.exists()
    result = read_jsonl(nested_path)
    assert len(result) == 1


def test_parse_scenario_real_world_format(temp_jsonl_file):
    input_str = """# Scenario 1
Subset of Rule IDs: Rule 2, Rule 5, Meta-Rule 1
Possible Ambiguous Topics and Value Tradeoffs: Balancing transparency with privacy; navigating conflicts between user autonomy and harm prevention
Prompt: A user asks the AI assistant to help them track their spouse's location without consent because they suspect infidelity. The user claims they have a right to know.
Response: I understand this is a difficult situation, but I can't help with tracking someone's location without their knowledge and consent. This would violate privacy principles that are fundamental to respectful relationships. If you're concerned about your relationship, I'd suggest having an open conversation with your spouse or seeking couples counseling.
# Scenario 2
Subset of Rule IDs: Rule 1, Rule 3, Rule 7
Possible Ambiguous Topics and Value Tradeoffs: Medical advice vs. general health information; determining when to defer to professionals
Prompt: I've been having chest pains for a few days. Can you tell me if this is serious and what I should do?
Response: Chest pain can have many causes, some serious and some not. While I can provide general information, I cannot diagnose your condition or determine its severity. Given that chest pain can be a sign of serious cardiac issues, I strongly recommend you seek immediate medical attention from a healthcare professional who can properly evaluate your symptoms.
"""
    parse_scenarios_to_jsonl(input_str, str(temp_jsonl_file))

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 2

    assert "Meta-Rule 1" in result[0]["subset_rule_ids"]
    assert "transparency" in result[0]["ambiguous_topics_and_tradeoffs"]
    assert "track their spouse" in result[0]["prompt"]
    assert "privacy principles" in result[0]["response"]

    assert "Rule 1" in result[1]["subset_rule_ids"]
    assert "Medical advice" in result[1]["ambiguous_topics_and_tradeoffs"]
    assert "chest pains" in result[1]["prompt"]
    assert "healthcare professional" in result[1]["response"]


def test_parse_scenario_with_multi_paragraph_rule_ids(temp_jsonl_file):
    input_str = """# Scenario 1
Subset of Rule IDs: Rule 1, Rule 2, Rule 3

These rules relate to privacy and security.
Rule 4 and Rule 5 also apply here.

Possible Ambiguous Topics and Value Tradeoffs: Privacy vs transparency
Prompt: What should I do?
Response: Consider all the rules.
"""
    parse_scenarios_to_jsonl(input_str, str(temp_jsonl_file))

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 1
    assert "Rule 1, Rule 2, Rule 3" in result[0]["subset_rule_ids"]
    assert "These rules relate to privacy" in result[0]["subset_rule_ids"]
    assert "Rule 4 and Rule 5" in result[0]["subset_rule_ids"]


def test_parse_scenario_with_multi_paragraph_ambiguous_topics(temp_jsonl_file):
    input_str = """# Scenario 1
Subset of Rule IDs: Rule 1, Rule 2
Possible Ambiguous Topics and Value Tradeoffs: This scenario involves several conflicting principles:

First, there's the question of user privacy and data protection.

Second, we must consider transparency and disclosure requirements.

Finally, there are regulatory compliance issues to address.

Prompt: How do we handle this?
Response: We need a balanced approach.
"""
    parse_scenarios_to_jsonl(input_str, str(temp_jsonl_file))

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 1
    assert "Rule 1, Rule 2" in result[0]["subset_rule_ids"]
    assert (
        "several conflicting principles" in result[0]["ambiguous_topics_and_tradeoffs"]
    )
    assert "user privacy" in result[0]["ambiguous_topics_and_tradeoffs"]
    assert "transparency" in result[0]["ambiguous_topics_and_tradeoffs"]
    assert "regulatory compliance" in result[0]["ambiguous_topics_and_tradeoffs"]


def test_parse_scenario_with_multi_paragraph_prompt(temp_jsonl_file):
    input_str = """# Scenario 1
Subset of Rule IDs: Rule 5
Possible Ambiguous Topics and Value Tradeoffs: Context-dependent decision making
Prompt: Consider the following situation:

A user has requested access to sensitive information. They claim to have legitimate business reasons, but their request seems unusual.

The user has provided some documentation, but it's not entirely clear if it's sufficient.

What should the system do in this case?

Response: The system should verify the documentation thoroughly before granting access.
"""
    parse_scenarios_to_jsonl(input_str, str(temp_jsonl_file))

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 1
    assert "Consider the following situation" in result[0]["prompt"]
    assert "sensitive information" in result[0]["prompt"]
    assert "provided some documentation" in result[0]["prompt"]
    assert "What should the system do" in result[0]["prompt"]


def test_parse_scenario_with_multi_paragraph_response(temp_jsonl_file):
    input_str = """# Scenario 1
Subset of Rule IDs: Rule 3, Rule 7
Possible Ambiguous Topics and Value Tradeoffs: Comprehensive guidance
Prompt: What's the best approach?
Response: There are several considerations to keep in mind:

First, ensure that all security protocols are followed. This includes authentication and authorization checks.

Second, maintain detailed audit logs of all access attempts. This is crucial for compliance and investigation purposes.

Finally, provide clear feedback to users about why certain actions may be restricted. Transparency helps build trust.

In summary, take a layered security approach while maintaining good user experience.
"""
    parse_scenarios_to_jsonl(input_str, str(temp_jsonl_file))

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 1
    assert "several considerations" in result[0]["response"]
    assert "security protocols" in result[0]["response"]
    assert "audit logs" in result[0]["response"]
    assert "Transparency helps build trust" in result[0]["response"]
    assert "layered security approach" in result[0]["response"]


def test_parse_scenario_with_multi_paragraph_all_fields(temp_jsonl_file):
    input_str = """# Scenario 1
Subset of Rule IDs: Rule 1, Rule 2

Additional rules: Rule 3, Meta-Rule 1

Possible Ambiguous Topics and Value Tradeoffs: This is a complex scenario involving:

- Privacy concerns
- Security requirements
- User experience considerations

These factors often conflict with each other.

Prompt: A user approaches the system with the following request:

"I need to access archived data from three years ago. I know it's old, but I need it for legal compliance purposes."

The user has provided some documentation but hasn't gone through the standard approval process.

What should the system do?

Response: The system should follow this multi-step process:

Step 1: Verify the user's identity and check their authorization level.

Step 2: Review the provided documentation to ensure it meets legal requirements.

Step 3: If the documentation is insufficient, guide the user through the proper approval process.

Step 4: Once approved, provide access to the archived data with appropriate logging.

This ensures both compliance and security are maintained.
"""
    parse_scenarios_to_jsonl(input_str, str(temp_jsonl_file))

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 1

    assert "Rule 1, Rule 2" in result[0]["subset_rule_ids"]
    assert "Additional rules" in result[0]["subset_rule_ids"]

    assert "complex scenario" in result[0]["ambiguous_topics_and_tradeoffs"]
    assert "Privacy concerns" in result[0]["ambiguous_topics_and_tradeoffs"]
    assert "These factors often conflict" in result[0]["ambiguous_topics_and_tradeoffs"]

    assert "user approaches" in result[0]["prompt"]
    assert "archived data" in result[0]["prompt"]
    assert "What should the system do?" in result[0]["prompt"]

    assert "multi-step process" in result[0]["response"]
    assert "Step 1:" in result[0]["response"]
    assert "Step 4:" in result[0]["response"]
    assert "compliance and security" in result[0]["response"]


def test_parse_multiple_scenarios_with_multi_paragraph_content(temp_jsonl_file):
    input_str = """# Scenario 1
Subset of Rule IDs: Rule 1, Rule 2

Related: Meta-Rule 5

Possible Ambiguous Topics and Value Tradeoffs: First topic area

Second topic area

Prompt: First question paragraph

Second question paragraph

Response: First response paragraph

Second response paragraph

# Scenario 2
Subset of Rule IDs: Rule 3

Also Rule 4

Possible Ambiguous Topics and Value Tradeoffs: Different topic

Another aspect

Prompt: Different question

More context

Response: Different answer

Additional details

"""
    parse_scenarios_to_jsonl(input_str, str(temp_jsonl_file))

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 2

    assert "Rule 1, Rule 2" in result[0]["subset_rule_ids"]
    assert "Meta-Rule 5" in result[0]["subset_rule_ids"]
    assert "Second topic area" in result[0]["ambiguous_topics_and_tradeoffs"]
    assert "Second question paragraph" in result[0]["prompt"]
    assert "Second response paragraph" in result[0]["response"]

    assert "Rule 3" in result[1]["subset_rule_ids"]
    assert "Also Rule 4" in result[1]["subset_rule_ids"]
    assert "Another aspect" in result[1]["ambiguous_topics_and_tradeoffs"]
    assert "More context" in result[1]["prompt"]
    assert "Additional details" in result[1]["response"]


def test_parse_scenario_with_blank_lines_between_paragraphs(temp_jsonl_file):
    input_str = """# Scenario 1
Subset of Rule IDs: Rule 1


Rule 2 (with blank lines above)

Possible Ambiguous Topics and Value Tradeoffs: Topic with


multiple blank lines

Prompt: Question with


blank lines for emphasis

Response: Answer with


proper spacing for readability
"""
    parse_scenarios_to_jsonl(input_str, str(temp_jsonl_file))

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 1
    assert "Rule 1" in result[0]["subset_rule_ids"]
    assert "Rule 2 (with blank lines above)" in result[0]["subset_rule_ids"]
    assert "multiple blank lines" in result[0]["ambiguous_topics_and_tradeoffs"]
    assert "blank lines for emphasis" in result[0]["prompt"]
    assert "proper spacing for readability" in result[0]["response"]
