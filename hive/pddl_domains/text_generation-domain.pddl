(define (domain text_generation)
	(:requirements :strips :typing)
	(:types  text)
	(:predicates
		(IsText ?input_text - text)
		(GenerateText ?input_text - text)
		(GenerateCode ?input_text - text)
	)
	(:action generate_or_write_text ; given a piece of text, generates a text
		:parameters (?input_text - text)
		:precondition (IsText ?input_text)
		:effect (GenerateText ?input_text)
	)
	(:action generate_or_write_code ; given a piece of text, generates code
		:parameters (?input_text - text)
		:precondition (IsText ?input_text)
		:effect (GenerateCode ?input_text)
	)
)
