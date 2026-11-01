import numpy as np

#####################################################
def create_qkv(X: np.ndarray,
                W_Q: np.ndarray,
                W_K: np.ndarray,
                W_V: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    #Checking dimensions
    assert X.ndim == 3, "X must have a shape of (B, T, d_model)"

    assert W_Q.ndim == 2, "W_Q must be a 2D matrix"
    assert W_K.ndim == 2, "W_K must be a 2D matrix"
    assert W_V.ndim == 2, "W_V must be a 2D matrix"


    B, T, d_model = X.shape

    assert W_Q.shape[0] == d_model, "W_Q input dimensions mismatch"
    assert W_K.shape[0] == d_model, "W_K input dimensions mismatch"
    assert W_V.shape[0] == d_model, "W_V input dimensions mismatch"

    assert W_Q.shape[1] == W_K.shape[1], "Q and K widths must match"

    Q = np.matmul(X, W_Q)
    K = np.matmul(X, W_K)
    V = np.matmul(X, W_V)

    return Q, K, V

#####################################################

def calculate_scores(Q: np.ndarray, K:np.ndarray) -> np.ndarray:
    assert Q.ndim == 3 and K.ndim == 3, "Q and K must have shape (B, T, d_k)"

    B, T, d_k = Q.shape

    assert K.shape == Q.shape, "Shape of Q and K must match"

    K_T = np.swapaxes(K, -1, -2)

    S = np.matmul(Q, K_T)  #dot-product attention scores

    S /= np.sqrt(d_k)

    return S
#####################################################


def create_causal_mask(T: int) -> np.ndarray:

    assert T > 0, "Sequence length must be positive"

    mask = np.triu(np.ones((T, T)), k = 1)
    mask = np.where(mask == 1, -np.inf, 0.0)

    return mask


def apply_causal_mask(S: np.ndarray) -> np.ndarray:
    assert S.ndim == 3, "S must have shape (B, T, T)"

    B, T, T2 = S.shape

    assert T == T2, "Attention score matrix must be square"

    mask = create_causal_mask(T)

    masked_S = S + mask

    return masked_S

#####################################################

def softmax(X:np.ndarray) -> np.ndarray:

    assert X.ndim == 3, "X must have shape (B, T, T)"

    row_max = np.max(X, axis=-1, keepdims=True)

    shifted_X = X - row_max

    exp_X = np.exp(shifted_X)

    denominator = np.sum(exp_X, axis=-1, keepdims=True)

    A = exp_X / denominator

    return A

#####################################################


def calculate_output(A:np.ndarray, V:np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    assert A.ndim == 3, "A must have shape (B, T, T)"
    assert V.ndim == 3, "V must have shape (B, T, d_k)"

    B, T, T2 = A.shape

    assert T == T2, "Attention score matrix must be square"
    assert B == V.shape[0] and T == V.shape[1]

    O = np.matmul(A, V)

    return O, A

#####################################################


def test_attention(X:np.ndarray, Q:np.ndarray, K:np.ndarray, V:np.ndarray, S:np.ndarray, masked_S:np.ndarray
                   , A:np.ndarray, O:np.ndarray) -> None:

    B, T, d_model = X.shape
    d_k = K.shape[-1]
    d_v = V.shape[-1]

    assert Q.shape == (B, T, d_k), "Q must have shape (B, T, d_k)"
    assert K.shape == (B, T, d_k), "K must have shape (B, T, d_k)"
    assert V.shape == (B, T, d_v), "V must have shape (B, T, d_k)"

    assert S.shape == (B, T, T), "S must have shape (B, T, T)"
    assert masked_S.shape == (B, T, T), "masked_S must have shape (B, T, T)"
    assert A.shape == (B, T, T), "A must have shape (B, T, T)"
    assert O.shape == (B, T, d_v), "O must have shape (B, T, d_v)"

    assert np.allclose(np.sum(A, axis=-1), 1.0), "Attention rows do not sum to 1"
    assert np.allclose(np.triu(A, k=1), 0.0), "Causal masking failed"
    assert np.all(np.isneginf(
    masked_S[:, np.triu_indices(T, k=1)[0],
             np.triu_indices(T, k=1)[1]]
)), "Masked scores are incorrect"

    assert np.all(np.isfinite(S)), "Scores contain NaN or infinity"
    assert np.all(np.isfinite(A)), "Attention contains NaN or infinity"
    assert np.all(np.isfinite(O)), "Output contains NaN or infinity"

    print("All attention tests passed.")
#####################################################


B = 1
T = 3
d_model = 4
d_k = 2
d_v = 2

X = np.array([[
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 1, 0, 0]
]], dtype=float)

W_Q = np.ones((d_model, d_k))
W_K = np.ones((d_model, d_k))
W_V = np.eye(d_model, d_v)

Q, K, V = create_qkv(X, W_Q, W_K, W_V)

S = calculate_scores(Q, K)

masked_S = apply_causal_mask(S)

A = softmax(masked_S)

O, A = calculate_output(A, V)

test_attention(X, Q, K, V, S, masked_S, A, O)

print("\nQ:\n", Q)
print("\nK:\n", K)
print("\nV:\n", V)
print("\nAttention weights:\n", A)
print("\nFinal output:\n", O)