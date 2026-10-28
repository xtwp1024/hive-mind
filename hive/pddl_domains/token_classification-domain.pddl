(define (domain token_classification)
	(:requirements :typing :strips)
	(:types text)
	(:predicates
		(IsText ?input_text - text)
		(GetNER ?input_text - text)
		(GetPOS ?input_text - text)
	)
	(:action get_named_entities ; Classify the recognized named entities from a text into predefined categories like persons, organizations, locations, etc.
		:parameters (?input_text - text)
		:precondition (IsText ?input_text)
		:effect (GetNER ?input_text)
	)
	(:action get_part_of_speech ; Classify each token into part of speech
		:parameters (?input_text - text)
		:precondition (IsText ?input_text)
		:effect (GetPOS ?input_text)
	)
)
