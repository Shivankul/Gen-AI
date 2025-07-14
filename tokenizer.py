import tiktoken
encoder = tiktoken.encoding_for_model('gpt-4o')
print("Vocab size" , encoder.n_vocab)
text = "the cat sat on the mat"
tokens = encoder.encode(text)
print("Tokens" , tokens)
my_tokens = [3086, 9059, 10139, 402, 290, 2450]
print(encoder.decode(my_tokens))