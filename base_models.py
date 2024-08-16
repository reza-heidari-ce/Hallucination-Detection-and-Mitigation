from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


def get_base_model(model_name):

    if model_name == "bert-base-multilingual-xquad":

        model_name_path = "alon-albalak/bert-base-multilingual-xquad"

        tokenizer = AutoTokenizer.from_pretrained(model_name_path)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name_path)
        base_model = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0) # Use device=0 for GPU

    return base_model