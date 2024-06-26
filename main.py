from simple_tokenizer import SimpleTokenizer

if __name__ == "__main__":
    tokenizer = SimpleTokenizer(vocab_size=400)
    training_text = "Large Language Models (LLMs) are a type of artificial intelligence (AI) designed to understand and generate human language in a coherent and contextually relevant manner. These models are built using deep learning techniques, particularly a branch known as transformer models, which have revolutionized the field of natural language processing (NLP). The core idea behind LLMs is to train a neural network on vast amounts of text data so that it can learn the statistical patterns of language and apply this knowledge to a wide range of language-related tasks.\nAt their heart, LLMs like OpenAI's GPT series, Google's BERT, and others, rely on massive datasets that encompass a diverse array of linguistic inputs. These datasets typically include books, websites, articles, and other text sources. By processing this extensive data, the models learn to predict the next word in a sentence, capture the meaning of words in context, and generate human-like text. The training process involves adjusting the weights of the neural network to minimize the difference between its predictions and the actual text in the training data.\nThe architecture of LLMs is based on the transformer model, introduced by Vaswani et al. in 2017. Transformers rely on a mechanism called attention, which allows the model to weigh the importance of different words in a sentence differently. This is crucial for understanding context and meaning, as it helps the model focus on relevant parts of the input text while ignoring less important information. Unlike previous models that processed language in a linear, sequential manner, transformers can handle multiple parts of a sentence simultaneously, making them more efficient and effective at capturing long-range dependencies.\nOne of the key breakthroughs of LLMs is their ability to perform a wide range of tasks without task-specific training. Once trained on general language data, these models can be fine-tuned for specific applications, such as translation, summarization, question answering, and more. This adaptability is due to the general-purpose nature of the language patterns they learn during training. By simply providing a model with examples of the desired task, it can adjust its internal representations to perform that task effectively.\nDespite their impressive capabilities, LLMs are not without limitations. One major challenge is their tendency to produce plausible-sounding but incorrect or nonsensical answers. This is because the models generate text based on patterns in the training data rather than an understanding of the world. Additionally, LLMs can inadvertently produce biased or harmful content if such biases are present in the training data. Researchers are actively working on methods to mitigate these issues, such as incorporating ethical guidelines into the training process and developing more robust evaluation metrics.\nAnother important consideration is the computational resources required to train and deploy LLMs. Training these models requires significant computational power, often involving specialized hardware like GPUs or TPUs and running for extended periods. This makes the development of LLMs resource-intensive and expensive, which can be a barrier to entry for smaller organizations and researchers. Furthermore, the deployment of these models also demands substantial computational resources, especially for real-time applications where latency and speed are critical.\nThe impact of LLMs on society is profound and multifaceted. They have the potential to transform industries by automating complex language tasks, enhancing productivity, and enabling new applications that were previously unimaginable. For instance, in healthcare, LLMs can assist in diagnosing diseases by analyzing medical records and literature. In customer service, they can provide instant, accurate responses to customer inquiries, improving satisfaction and efficiency. In education, they can serve as personalized tutors, helping students learn and understand complex subjects at their own pace.\nHowever, the widespread use of LLMs also raises important ethical and societal questions. The potential for misuse, such as generating fake news or deepfakes, is a significant concern. There is also the issue of job displacement, as automation powered by LLMs could replace certain types of work. Ensuring that the benefits of these technologies are distributed equitably and that their deployment is governed by ethical principles is a critical challenge for policymakers, technologists, and society at large.\nIn summary, Large Language Models represent a significant advancement in the field of artificial intelligence, offering powerful tools for understanding and generating human language. Their development and deployment come with both immense potential and significant challenges. As the field continues to evolve, addressing the ethical, societal, and technical issues associated with LLMs will be crucial to harnessing their benefits while mitigating their risks."
    tokenizer.train(training_text)

    # Now you can use the tokenizer to encode and decode text
    encoded = tokenizer.encode("hey everyone, this test is tokenized using a simple tokenizer from scratch!")
    print(encoded)
    decoded = tokenizer.decode(encoded)
    print(decoded)