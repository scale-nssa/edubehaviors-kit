# Assertions

The table below shows the 49 assertions usable by `AssertionAnnotator` and `ClassificationPipeline`.
Each assertion has a corresponding SetFit classifier on HuggingFace.
The labeled training dataset is also available on HuggingFace at [`StanfordSCALE/assertions_llm_annotated_talkmoves`](https://huggingface.co/datasets/StanfordSCALE/assertions_llm_annotated_talkmoves).

!!!warning "Model performance varies"
    All available assertions are listed here, but some assertions have poor performance or
    poor inter-rater reliability. Consider assertion performance before using annotators to
    make claims about your data.

| Assertion | Model | Krippendorff's alpha | Test F1 |
| --- | --- | ---: | ---: |
| `sentence_has_negation_or_denial` | [StanfordSCALE/assertion_sentence_has_negation_or_denial](https://huggingface.co/StanfordSCALE/assertion_sentence_has_negation_or_denial) | 0.902 | 0.941 |
| `sentence_includes_student_name` | [StanfordSCALE/assertion_sentence_includes_student_name](https://huggingface.co/StanfordSCALE/assertion_sentence_includes_student_name) | 0.930 | 0.933 |
| `sentence_has_a_question` | [StanfordSCALE/assertion_sentence_has_a_question](https://huggingface.co/StanfordSCALE/assertion_sentence_has_a_question) | 0.867 | 0.922 |
| `sentence_has_math_terms` | [StanfordSCALE/assertion_sentence_has_math_terms](https://huggingface.co/StanfordSCALE/assertion_sentence_has_math_terms) | 0.755 | 0.900 |
| `sentence_calls_on_student_by_name` | [StanfordSCALE/assertion_sentence_calls_on_student_by_name](https://huggingface.co/StanfordSCALE/assertion_sentence_calls_on_student_by_name) | 0.918 | 0.889 |
| `sentence_is_a_declarative_statement_or_description` | [StanfordSCALE/assertion_sentence_is_a_declarative_statement_or_description](https://huggingface.co/StanfordSCALE/assertion_sentence_is_a_declarative_statement_or_description) | 0.815 | 0.885 |
| `sentence_has_number` | [StanfordSCALE/assertion_sentence_has_number](https://huggingface.co/StanfordSCALE/assertion_sentence_has_number) | 0.755 | 0.874 |
| `sentence_has_fraction_terms` | [StanfordSCALE/assertion_sentence_has_fraction_terms](https://huggingface.co/StanfordSCALE/assertion_sentence_has_fraction_terms) | 0.876 | 0.868 |
| `sentence_has_acknowledgment` | [StanfordSCALE/assertion_sentence_has_acknowledgment](https://huggingface.co/StanfordSCALE/assertion_sentence_has_acknowledgment) | 0.709 | 0.840 |
| `sentence_has_agreement_or_affirmation` | [StanfordSCALE/assertion_sentence_has_agreement_or_affirmation](https://huggingface.co/StanfordSCALE/assertion_sentence_has_agreement_or_affirmation) | 0.586 | 0.801 |
| `sentence_has_apology` | [StanfordSCALE/assertion_sentence_has_apology](https://huggingface.co/StanfordSCALE/assertion_sentence_has_apology) | 0.869 | 0.800 |
| `sentence_addresses_the_whole_class` | [StanfordSCALE/assertion_sentence_addresses_the_whole_class](https://huggingface.co/StanfordSCALE/assertion_sentence_addresses_the_whole_class) | 0.546 | 0.799 |
| `sentence_has_comparison_terms` | [StanfordSCALE/assertion_sentence_has_comparison_terms](https://huggingface.co/StanfordSCALE/assertion_sentence_has_comparison_terms) | 0.657 | 0.796 |
| `sentence_has_a_directive_or_instruction` | [StanfordSCALE/assertion_sentence_has_a_directive_or_instruction](https://huggingface.co/StanfordSCALE/assertion_sentence_has_a_directive_or_instruction) | 0.735 | 0.790 |
| `sentence_is_a_short_utterance` | [StanfordSCALE/assertion_sentence_is_a_short_utterance](https://huggingface.co/StanfordSCALE/assertion_sentence_is_a_short_utterance) | 0.438 | 0.789 |
| `sentence_has_praise_or_encouragement` | [StanfordSCALE/assertion_sentence_has_praise_or_encouragement](https://huggingface.co/StanfordSCALE/assertion_sentence_has_praise_or_encouragement) | 0.604 | 0.780 |
| `sentence_uses_collaborative_or_inclusive_language` | [StanfordSCALE/assertion_sentence_uses_collaborative_or_inclusive_language](https://huggingface.co/StanfordSCALE/assertion_sentence_uses_collaborative_or_inclusive_language) | 0.559 | 0.765 |
| `sentence_has_politeness_marker` | [StanfordSCALE/assertion_sentence_has_politeness_marker](https://huggingface.co/StanfordSCALE/assertion_sentence_has_politeness_marker) | 0.731 | 0.760 |
| `sentence_invites_participation` | [StanfordSCALE/assertion_sentence_invites_participation](https://huggingface.co/StanfordSCALE/assertion_sentence_invites_participation) | 0.610 | 0.750 |
| `sentence_has_counting_sequence` | [StanfordSCALE/assertion_sentence_has_counting_sequence](https://huggingface.co/StanfordSCALE/assertion_sentence_has_counting_sequence) | 0.757 | 0.732 |
| `sentence_has_time_reference` | [StanfordSCALE/assertion_sentence_has_time_reference](https://huggingface.co/StanfordSCALE/assertion_sentence_has_time_reference) | 0.603 | 0.728 |
| `sentence_references_classroom_materials_or_visuals` | [StanfordSCALE/assertion_sentence_references_classroom_materials_or_visuals](https://huggingface.co/StanfordSCALE/assertion_sentence_references_classroom_materials_or_visuals) | 0.686 | 0.719 |
| `sentence_references_task_procedure_or_logistics` | [StanfordSCALE/assertion_sentence_references_task_procedure_or_logistics](https://huggingface.co/StanfordSCALE/assertion_sentence_references_task_procedure_or_logistics) | 0.404 | 0.703 |
| `sentence_references_student_behavior_or_work` | [StanfordSCALE/assertion_sentence_references_student_behavior_or_work](https://huggingface.co/StanfordSCALE/assertion_sentence_references_student_behavior_or_work) | 0.382 | 0.672 |
| `sentence_checks_for_understanding_or_agreement` | [StanfordSCALE/assertion_sentence_checks_for_understanding_or_agreement](https://huggingface.co/StanfordSCALE/assertion_sentence_checks_for_understanding_or_agreement) | 0.324 | 0.664 |
| `sentence_has_explanation_or_reasoning` | [StanfordSCALE/assertion_sentence_has_explanation_or_reasoning](https://huggingface.co/StanfordSCALE/assertion_sentence_has_explanation_or_reasoning) | 0.593 | 0.615 |
| `sentence_expresses_personal_stance_or_thinking_aloud` | [StanfordSCALE/assertion_sentence_expresses_personal_stance_or_thinking_aloud](https://huggingface.co/StanfordSCALE/assertion_sentence_expresses_personal_stance_or_thinking_aloud) | 0.447 | 0.579 |
| `sentence_manages_classroom_behavior_or_attention` | [StanfordSCALE/assertion_sentence_manages_classroom_behavior_or_attention](https://huggingface.co/StanfordSCALE/assertion_sentence_manages_classroom_behavior_or_attention) | 0.460 | 0.566 |
| `sentence_poses_a_hypothetical_or_scenario` | [StanfordSCALE/assertion_sentence_poses_a_hypothetical_or_scenario](https://huggingface.co/StanfordSCALE/assertion_sentence_poses_a_hypothetical_or_scenario) | 0.373 | 0.550 |
| `sentence_has_measurement_terms` | [StanfordSCALE/assertion_sentence_has_measurement_terms](https://huggingface.co/StanfordSCALE/assertion_sentence_has_measurement_terms) | 0.180 | 0.548 |
| `sentence_quotes_or_reads_text_aloud` | [StanfordSCALE/assertion_sentence_quotes_or_reads_text_aloud](https://huggingface.co/StanfordSCALE/assertion_sentence_quotes_or_reads_text_aloud) | 0.661 | 0.529 |
| `sentence_seeks_or_gives_clarification` | [StanfordSCALE/assertion_sentence_seeks_or_gives_clarification](https://huggingface.co/StanfordSCALE/assertion_sentence_seeks_or_gives_clarification) | 0.270 | 0.506 |
| `sentence_narrates_ongoing_action` | [StanfordSCALE/assertion_sentence_narrates_ongoing_action](https://huggingface.co/StanfordSCALE/assertion_sentence_narrates_ongoing_action) | 0.594 | 0.495 |
| `sentence_has_informal_language` | [StanfordSCALE/assertion_sentence_has_informal_language](https://huggingface.co/StanfordSCALE/assertion_sentence_has_informal_language) | -0.120 | 0.494 |
| `sentence_has_answer_to_a_math_problem` | [StanfordSCALE/assertion_sentence_has_answer_to_a_math_problem](https://huggingface.co/StanfordSCALE/assertion_sentence_has_answer_to_a_math_problem) | 0.409 | 0.464 |
| `sentence_shows_uncertainty` | [StanfordSCALE/assertion_sentence_shows_uncertainty](https://huggingface.co/StanfordSCALE/assertion_sentence_shows_uncertainty) | 0.503 | 0.459 |
| `sentence_expresses_certainty_or_emphasis` | [StanfordSCALE/assertion_sentence_expresses_certainty_or_emphasis](https://huggingface.co/StanfordSCALE/assertion_sentence_expresses_certainty_or_emphasis) | 0.195 | 0.441 |
| `sentence_expresses_emotion_or_humor` | [StanfordSCALE/assertion_sentence_expresses_emotion_or_humor](https://huggingface.co/StanfordSCALE/assertion_sentence_expresses_emotion_or_humor) | 0.275 | 0.421 |
| `sentence_references_prior_learning_or_lesson` | [StanfordSCALE/assertion_sentence_references_prior_learning_or_lesson](https://huggingface.co/StanfordSCALE/assertion_sentence_references_prior_learning_or_lesson) | 0.442 | 0.393 |
| `sentence_evaluates_a_student_response` | [StanfordSCALE/assertion_sentence_evaluates_a_student_response](https://huggingface.co/StanfordSCALE/assertion_sentence_evaluates_a_student_response) | 0.433 | 0.391 |
| `sentence_has_disagreement_or_challenge` | [StanfordSCALE/assertion_sentence_has_disagreement_or_challenge](https://huggingface.co/StanfordSCALE/assertion_sentence_has_disagreement_or_challenge) | 0.166 | 0.332 |
| `sentence_repeats_or_revoices_prior_speech` | [StanfordSCALE/assertion_sentence_repeats_or_revoices_prior_speech](https://huggingface.co/StanfordSCALE/assertion_sentence_repeats_or_revoices_prior_speech) | 0.419 | 0.313 |
| `sentence_is_incomplete_or_trails_off` | [StanfordSCALE/assertion_sentence_is_incomplete_or_trails_off](https://huggingface.co/StanfordSCALE/assertion_sentence_is_incomplete_or_trails_off) | 0.369 | 0.292 |
| `sentence_shows_realization_or_insight` | [StanfordSCALE/assertion_sentence_shows_realization_or_insight](https://huggingface.co/StanfordSCALE/assertion_sentence_shows_realization_or_insight) | 0.080 | 0.229 |
| `sentence_answers_a_question` | [StanfordSCALE/assertion_sentence_answers_a_question](https://huggingface.co/StanfordSCALE/assertion_sentence_answers_a_question) | 0.101 | 0.218 |
| `sentence_has_a_rhetorical_question` | [StanfordSCALE/assertion_sentence_has_a_rhetorical_question](https://huggingface.co/StanfordSCALE/assertion_sentence_has_a_rhetorical_question) | 0.180 | 0.176 |
| `sentence_expresses_confusion_or_requests_help` | [StanfordSCALE/assertion_sentence_expresses_confusion_or_requests_help](https://huggingface.co/StanfordSCALE/assertion_sentence_expresses_confusion_or_requests_help) | 0.218 | 0.160 |
| `sentence_summarizes_or_reviews` | [StanfordSCALE/assertion_sentence_summarizes_or_reviews](https://huggingface.co/StanfordSCALE/assertion_sentence_summarizes_or_reviews) | 0.229 | 0.128 |
| `sentence_grants_or_requests_permission` | [StanfordSCALE/assertion_sentence_grants_or_requests_permission](https://huggingface.co/StanfordSCALE/assertion_sentence_grants_or_requests_permission) | 0.500 | 0.118 |

**Krippendorff's alpha** is the agreement between the two LLM annotators that labelled the 3,217 utterances in the dataset.
**Test F1** is the positive-class F1 of the published classifier on the held-out test set.
