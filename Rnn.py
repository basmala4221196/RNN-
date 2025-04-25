import numpy as np

words = ["I", "am", "very", "happy"]
vocab_size = len(words)  
hidden_size = 3  
input_sequence = words[:3] 
target_word = words[3]  


word_to_index = {word: i for i, word in enumerate(words)}
one_hot = {word: np.zeros(vocab_size) for word in words}
for word in one_hot:
    one_hot[word][word_to_index[word]] = 1
    

np.random.seed(42)
Wx = np.random.randn(hidden_size, vocab_size) * 0.01  
Wh = np.random.randn(hidden_size, hidden_size) * 0.01 
Wy = np.random.randn(vocab_size, hidden_size) * 0.01   

def forward_propagation(inputs, h_prev):
  
  
    xs, hs, ats, ys = {}, {}, {}, {}
    hs[-1] = np.copy(h_prev)  
    
    for t in range(len(inputs)):
        xs[t] = one_hot[inputs[t]].reshape(vocab_size, 1)  # One-Hot Vector (k x 1)
        ats[t] = np.dot(Wh, hs[t-1]) + np.dot(Wx, xs[t])  # a_t = W_h h_{t-1} + W_x x_t (d x 1)
        hs[t] = np.tanh(ats[t])  # h_t = tanh(a_t) (d x 1)
        ys[t] = np.dot(Wy, hs[t])  # y_t = W_y h_t (k x 1)
        ys[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  #(k x 1)
    
    return xs, hs, ats, ys


def compute_loss(ys, target):
    target_idx = word_to_index[target]

    loss = -np.log(ys[len(ys)-1][target_idx, 0])
    return loss


def backward_propagation(xs, hs, ats, ys, target):
    dWx, dWh, dWy = np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(Wy)
    dh_next = np.zeros((hidden_size, 1))  
    
    target_idx = word_to_index[target]
    dy = np.copy(ys[len(ys)-1])  
    dy[target_idx] -= 1 
    
    for t in reversed(range(len(xs))):
        
        dWy += np.dot(dy, hs[t].T) 
        
        dh = np.dot(Wy.T, dy) + dh_next  
        dh_raw = (1 - hs[t] * hs[t]) * dh  
        
        dWx += np.dot(dh_raw, xs[t].T) 
        if t > 0:
            dWh += np.dot(dh_raw, hs[t-1].T)  
        
        dh_next = np.dot(Wh.T, dh_raw)  
        
        
        if t < len(xs) - 1:
            dy = np.zeros_like(dy)
            
    
    for dparam in [dWx, dWh, dWy]:
        np.clip(dparam, -5, 5, out=dparam)
    
    return dWx, dWh, dWy

def train(inputs, target, epochs=100, learning_rate=0.1):
    global Wx, Wh, Wy
    h_prev = np.zeros((hidden_size, 1))  
    
    for epoch in range(epochs):

        xs, hs, ats, ys = forward_propagation(inputs, h_prev)
        
        loss = compute_loss(ys, target)
        
        dWx, dWh, dWy = backward_propagation(xs, hs, ats, ys, target)
        
        Wx -= learning_rate * dWx
        Wh -= learning_rate * dWh
        Wy -= learning_rate * dWy
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return ys

def predict(inputs):
    h_prev = np.zeros((hidden_size, 1))
    xs, hs, ats, ys = forward_propagation(inputs, h_prev)
    last_y = ys[len(inputs)-1]
    predicted_idx = np.argmax(last_y)
    return words[predicted_idx]


print("Training RNN...")
ys = train(input_sequence, target_word, epochs=100, learning_rate=0.1)


predicted_word = predict(input_sequence)
print(f"\nInput Sequence: {input_sequence}")
print(f"Predicted Word: {predicted_word}")
print(f"Actual Word: {target_word}")
