import replicate
import os


class Mistral:
    def __init__(self, api_key):
        self.model_name = "mistralai/mistral-7b-instruct-v0.1:83b6a56e7c828e667f21fd596c338fd4f0039b46bcfa18d973e8e70e455fda70"
        self.debug = False
        self.top_k = 50
        self.top_p = 0.9
        self.temperature = 0.7
        self.system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        self.max_new_tokens = 500
        self.min_new_tokens = -1
        os.environ["REPLICATE_API_TOKEN"] = api_key

    def call_llm(self, prompt):
        output = replicate.run(
            self.model_name,
            input={
                "debug": self.debug,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "prompt": prompt,
                "temperature": self.temperature,
                "max_new_tokens": self.max_new_tokens,
                "min_new_tokens": self.min_new_tokens,
            },
        )

        res = []
        for o in output:
            res.append(o)

        return "".join(res).splitlines()

    def get_questions(self, caption, num_questions=5):
        prompt = f"Given the following caption, generate {num_questions} questions about the picture that you can ask a Visual Question and Answer Model. |{caption}|"

        question_list = self.call_llm(prompt)
        # print(question_list)
        res = []
        for q in question_list:
            if len(q) == 0:
                continue
            if q[-1] != "?":
                continue
            res.append(q.strip()[3:])
        return res

    def get_complete_summary(self, qna_list):
        prompt = f"Given the following questions and answers, generate a detailed summary of the image. |{' '.join(qna_list)}|"
        respone = self.call_llm(prompt)
        return respone


if __name__ == "__main__":
    mistral = Mistral()
    caption = "A dog playing with a football in a field"
    questions = mistral.get_questions(caption, num_questions=5)
    print(questions)
