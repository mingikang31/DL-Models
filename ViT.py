'''ViT Model with Conv2d, ConvNN, ConvNN_Attn, Attention'''

# Torch Imports 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchsummary import summary 
import numpy as np 



from typing import cast, Union

'''VGG Model Class'''
class ViT(nn.Module): 
    def __init__(self, args): 
        super(ViT, self).__init__()
        assert args.img_size[1] % args.patch_size == 0 and args.img_size[2] % args.patch_size == 0, "img_size dimensions must be divisible by patch_size dimensions"
        assert args.d_model % args.num_heads == 0, "d_model must be divisible by n_heads"
        
        self.args = args
        self.args.model = "VIT"
        self.model = "VIT"
        
        self.d_model = self.args.d_model # Dimensionality of Model
        self.img_size = self.args.img_size[1:]
        self.n_classes = self.args.num_classes # Number of Classes
        self.n_heads = self.args.num_heads
        self.patch_size = (self.args.patch_size, self.args.patch_size) # Patch Size
        self.n_channels = self.args.img_size[0]
        self.n_layers = self.args.num_layers # Number of Layers
        
        self.n_patches = (self.img_size[0] * self.img_size[1]) // (self.patch_size[0] * self.patch_size[1])
        
        
        self.dropout = self.args.dropout # Dropout Rate
        self.max_seq_length = self.n_patches + 1 # +1 for class token
        
        
        self.patch_embedding = PatchEmbedding(self.d_model, self.img_size, self.patch_size, self.n_channels) # Patch Embedding Layer
        self.positional_encoding = PositionalEncoding(self.d_model, self.max_seq_length)
        
        self.transformer_encoder = nn.Sequential(*[TransformerEncoder(args, self.d_model, self.n_heads) for _ in range(self.n_layers)])
        
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.n_classes)
        )
        
        self.device = args.device
        
        self.to(self.device)
        self.name = f"{self.args.model} {self.args.layer}"
        
    def forward(self, x): 
        x = self.patch_embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0]) # Taking the CLS token for classification
        return x

    def summary(self): 
        original_device = next(self.parameters()).device
        try:
            self.to("cpu")
            print(f"--- Summary for {self.name} ---")
            # torchsummary expects batch dimension, but img_size doesn't include it
            summary(self, input_size=self.img_size, device="cpu") 
        except Exception as e:
            print(f"Could not generate summary: {e}")
        finally:
            # Move model back to its original device
            self.to(original_device)
        
    def parameter_count(self): 
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
                
        
        
class PatchEmbedding(nn.Module): 
    def __init__(self, d_model, img_size, patch_size, n_channels=3): 
        super(PatchEmbedding, self).__init__()
        
        self.d_model = d_model # Dimensionality of Model 
        self.img_size = img_size # Size of Image
        self.patch_size = patch_size # Patch Size 
        self.n_channels = n_channels # Number of Channels in Image
        
        self.linear_projection = nn.Conv2d(in_channels=n_channels, out_channels=d_model, kernel_size=patch_size, stride=patch_size) # Linear Projection Layer
        self.norm = nn.LayerNorm(d_model) # Normalization Layer
        
        self.flatten = nn.Flatten(start_dim=2)
        
    def forward(self, x): 
        x = self.linear_projection(x) # (B, C, H, W) -> (B, d_model, H', W')
        x = self.flatten(x) # (B, d_model, H', W') -> (B, d_model, n_patches)
        x = x.transpose(1, 2) # (B, d_model, n_patches) -> (B, n_patches, d_model)
        x = self.norm(x) # (B, n_patches, d_model) -> (B, n_patches, d_model)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length): 
        super(PositionalEncoding, self).__init__()
        
        self.cls_tokens = nn.Parameter(torch.randn(1, 1, d_model)) # Classification Token
        
        # Creating Positional Encoding 
        pe = torch.zeros(max_seq_length, d_model)
        
        for pos in range(max_seq_length):
            for i in range(d_model):
                if i % 2 == 0:
                    pe[pos][i] = np.sin(pos/(10000 ** (i/d_model)))
                else:
                    pe[pos][i] = np.cos(pos/(10000 ** ((i-1)/d_model)))

        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x): 
        # Expand to have class token for each image in batch 
        tokens_batch = self.cls_tokens.expand(x.shape[0], -1, -1) # (B, 1, d_model)
        
        # Concatenate class token with positional encoding
        x = torch.cat((tokens_batch, x), dim=1)
        
        # Add positional encoding to the input 
        x = x + self.pe
        
        return x

class TransformerEncoder(nn.Module): 
    def __init__(self, args, d_model, num_heads, r_mlp=4, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.args = args 
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.r_mlp = r_mlp        
        self.dropout = dropout
        
        if args.layer == "Attention":
            self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        elif args.layer == "ConvNN": 
            self.attention = MultiHeadConvNN(d_model, num_heads, args.K, args.num_samples, args.magnitude_type)
        elif args.layer == "ConvNNAttention": 
            self.attention = MultiHeadConvNNAttention(d_model, num_heads, args.K, args.num_samples, args.magnitude_type)
        elif args.layer == "Conv1d":
            self.attention = MultiHeadConv1d(d_model, num_heads, args.kernel_size)
        elif args.layer == "Conv1dAttention":
            self.attention = MultiHeadConv1dAttention(d_model, num_heads, args.kernel_size)

        
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Multilayer Perceptron 
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * r_mlp),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * r_mlp, d_model)
        )
        
    def forward(self, x): 
        # Multi-Head Attention
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout1(attn_output))  
        
        # Feed Forward Network 
        mlp_output = self.mlp(x)
        x = self.norm2(x + self.dropout2(mlp_output)) 
        return x



"""Multi-Head Layers for Transformer Encoder"""
class MultiHeadAttention(nn.Module): 
    def __init__(self, d_model, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # dimension of each head
        self.dropout = nn.Dropout(dropout)
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)        
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = self.dropout(torch.softmax(attn_scores, dim=-1))
        output = torch.matmul(attn_probs, V)
        return output, attn_probs
    
    def split_head(self, x): 
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) # (B, num_heads, seq_length, d_k)
        
    def combine_heads(self, x): 
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model) 
    
    def forward(self, x, mask=None):
        q = self.split_head(self.W_q(x)) # (B, num_heads, seq_length, d_k)
        k = self.split_head(self.W_k(x))
        v = self.split_head(self.W_v(x))
        
        attn_output, _ = self.scaled_dot_product_attention(q, k, v, mask) # (B, num_heads, seq_length, d_k)
        output = self.W_o(self.combine_heads(attn_output)) # (B, seq_length, d_model)
        return output

class MultiHeadConvNNAttention(nn.Module):
    def __init__(self, d_model, num_heads, K, samples, magnitude_type):
        super(MultiHeadConvNNAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.K = K
        self.samples = int(samples) if samples != 0 else None
        self.magnitude_type = magnitude_type
        self.maximum = True if self.magnitude_type == 'similarity' else False
        
        # Linear projections for query, key, value
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)   
        
        
        self.in_channels = d_model // num_heads
        self.out_channels = d_model // num_heads
        self.kernel_size = K
        self.stride = K
        
        self.conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,
        )
        
    def split_head(self, x): 
        batch_size, seq_length, d_model = x.size()
        self.batch_size = batch_size
        self.seq_length = seq_length
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) # (B, num_heads, seq_length, d_k)
        
    def combine_heads(self, x): 
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model) 
    
    def batch_split(self, x): 
        x = x.reshape(self.batch_size, -1, self.d_k, self.seq_length)
        return x.permute(0, 1, 3, 2).contiguous()
        
    def batch_combine(self, x): 
        batch_size, _, seq_length, d_k = x.size()
        x = x.permute(0, 1, 3, 2).contiguous() 
        return x.view(-1, self.d_k, seq_length)
        
    def forward(self, x):
        if self.samples is None: # All Samples
            q = self.batch_combine(self.split_head(self.W_q(x)))
            k = self.batch_combine(self.split_head(self.W_k(x)))
            v = self.batch_combine(self.split_head(self.W_v(x)))
            
            
            # Calculate Distance/Similarity Matrix + Prime Vmap 2D
            if self.magnitude_type == 'distance': 
                matrix_magnitude = self._calculate_distance_matrix(k, q, sqrt=True)
            elif self.magnitude_type == 'similarity':
                matrix_magnitude = self._calculate_similarity_matrix(k, q)
                
            prime = self._prime(v, matrix_magnitude, self.K, self.maximum) 
            x = self.conv(prime)  
            
            x = self.W_o(self.combine_heads(self.batch_split(x.permute(0, 2, 1))))
      
            return x
        
        else: # Random Samples
            q = self.batch_combine(self.split_head(self.W_q(x)))
            k = self.batch_combine(self.split_head(self.W_k(x)))
            v = self.batch_combine(self.split_head(self.W_v(x)))

            # Calculate Distance/Similarity Matrix + Prime       
            rand_idx = torch.randperm(q.shape[2], device=q.device)[:self.samples]
            
            q_sample = q[:, :, rand_idx]
            
            if self.magnitude_type == 'distance':
                matrix_magnitude = self._calculate_distance_matrix_N(k, q_sample, sqrt=True)
            elif self.magnitude_type == 'similarity':
                matrix_magnitude = self._calculate_similarity_matrix_N(k, q_sample)
                
            range_idx = torch.arange(len(rand_idx), device=q.device)
                
        
            if self.magnitude_type == 'distance':
                matrix_magnitude[:, rand_idx, range_idx] = float('inf') 
            elif self.magnitude_type == 'similarity':
                matrix_magnitude[:, rand_idx, range_idx] = float('-inf')
            
            
            prime = self._prime_N(v, matrix_magnitude, self.K, rand_idx, self.maximum)
            
            # Conv1d Layer
            x = self.conv(prime)  
            
            x = self.W_o(self.combine_heads(self.batch_split(x.permute(0, 2, 1))))
      
            return x        

    def _calculate_similarity_matrix(self, K, Q):
        k_norm = F.normalize(K, p=2, dim=1)
        q_norm = F.normalize(Q, p=2, dim=1)
        similarity_matrix = torch.bmm(k_norm.transpose(2, 1), q_norm)  # [B, N, M]
        similarity_matrix = torch.clamp(similarity_matrix, min=0)  # remove negative values
        return similarity_matrix
    
    def _calculate_similarity_matrix_N(self, K, Q):
        k_norm = F.normalize(K, p=2, dim=1)
        q_norm = F.normalize(Q, p=2, dim=1)
        similarity_matrix = torch.bmm(k_norm.transpose(2, 1), q_norm)  # [B, N, M]
        similarity_matrix = torch.clamp(similarity_matrix, min=0)  # remove negative values
        return similarity_matrix
        

    def _calculate_distance_matrix(self, K, Q, sqrt=False):
        norm_squared_K = torch.sum(K**2, dim=1, keepdim=True) 
        norm_squared_Q = torch.sum(Q**2, dim=1, keepdim=True) 
        
        dot_product = torch.bmm(K.transpose(2, 1), Q)  
        
        dist_matrix = norm_squared_K + norm_squared_Q.transpose(2, 1) - 2 * dot_product
        
        dist_matrix = torch.clamp(dist_matrix, min=0)  # remove negative values
        
        if sqrt:
            dist_matrix = torch.sqrt(dist_matrix)
        
        return dist_matrix

    def _calculate_distance_matrix_N(self, K, Q, sqrt=False):
        norm_squared_K = torch.sum(K**2, dim=1, keepdim=True).permute(0, 2, 1)
        norm_squared_Q = torch.sum(Q**2, dim=1, keepdim=True).transpose(2, 1).permute(0, 2, 1)
        
        dot_product = torch.bmm(K.transpose(2, 1), Q)  
        
        # Broadcasting: [B, 1, N] + [B, M, 1] - 2*[B, N, M]
        dist_matrix = norm_squared_K + norm_squared_Q - 2 * dot_product
        
        dist_matrix = torch.clamp(dist_matrix, min=0)  # remove negative values
        
        if sqrt:
            dist_matrix = torch.sqrt(dist_matrix)
        
        return dist_matrix

    def _prime(self, v, qk, K, maximum):
        b, c, t = v.shape 
        
        _, topk_indices = torch.topk(qk, k=K, dim=-1, largest = maximum)
        topk_indices_exp = topk_indices.unsqueeze(1).expand(b, c, t, K)
        
        v_expanded = v.unsqueeze(-1).expand(b, c, t, K)
        
        prime = torch.gather(v_expanded, dim=2, index=topk_indices_exp)
        prime = prime.reshape(b, c, -1)

        return prime

    def _prime_N(self, v, qk, K, rand_idx, maximum):
        b, c, t = v.shape

        _, topk_indices = torch.topk(qk, k=K - 1, dim=2, largest=maximum)
        tk = topk_indices.shape[-1]
        assert K == tk + 1, "Error: K must be same as tk + 1. K == tk + 1."

        mapped_tensor = rand_idx[topk_indices]

        token_indices = torch.arange(t, device=v.device).view(1, t, 1).expand(b, t, 1)

        final_indices = torch.cat([token_indices, mapped_tensor], dim=2)

        indices_expanded = final_indices.unsqueeze(1).expand(b, c, t, K)

        v_expanded = v.unsqueeze(-1).expand(b, c, t, K).contiguous()

        prime = torch.gather(v_expanded, dim=2, index=indices_expanded)

        prime = prime.reshape(b, c, -1)
        return prime

class MultiHeadConvNN(nn.Module):
    def __init__(self, d_model, num_heads, K, samples, magnitude_type, seq_length=197):
        super(MultiHeadConvNN, self).__init__() 
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"   
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.K = K
        self.samples = int(samples) if samples != 0 else None
        self.magnitude_type = magnitude_type
        self.maximum = True if self.magnitude_type == 'similarity' else False
        self.seq_length = seq_length
        
        # Linear projections for query, key, value
        self.W_q = nn.Linear(self.seq_length, self.seq_length)
        self.W_k = nn.Linear(self.seq_length, self.seq_length)
        self.W_v = nn.Linear(self.seq_length, self.seq_length)
        self.W_o = nn.Linear(self.seq_length, self.seq_length)
        
        self.in_channels = d_model // num_heads
        self.out_channels = d_model // num_heads
        self.kernel_size = K
        self.stride = K
        self.padding = 0 
        
        self.conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

    def split_head(self, x):
        batch_size, d_model, seq_length = x.size()
        self.batch_size = batch_size
        self.seq_length = seq_length
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) # (B, num_heads, seq_length, d_k)
    
    def combine_heads(self, x): 
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, self.d_model, seq_length) 
    
    def batch_split(self, x): 
        x = x.reshape(self.batch_size, -1, self.d_k, self.seq_length)
        return x.permute(0, 1, 3, 2).contiguous()
        
    def batch_combine(self, x): 
        batch_size, _, seq_length, d_k = x.size()
        x = x.permute(0, 1, 3, 2).contiguous() 
        return x.view(-1, self.d_k, seq_length)

    def forward(self, x):
        if self.samples is None: # All Samples 
            x = x.permute(0, 2, 1) 
        
            q = self.batch_combine(self.split_head(self.W_q(x)))
            k = self.batch_combine(self.split_head(self.W_k(x)))
            v = self.batch_combine(self.split_head(self.W_v(x)))
       
            # Calculate Distance/Similarity Matrix + Prime Vmap 2D
            if self.magnitude_type == 'distance': 
                matrix_magnitude = self._calculate_distance_matrix(k, q, sqrt=True)
            elif self.magnitude_type == 'similarity':
                matrix_magnitude = self._calculate_similarity_matrix(k, q)
                
            prime = self._prime(v, matrix_magnitude, self.K, self.maximum) 
            x = self.conv(prime)  
            
            x = self.W_o(self.combine_heads(self.batch_split(x))).permute(0, 2, 1)
            return x
        else: # Random Samples  
            x = x.permute(0, 2, 1) 
        
            q = self.batch_combine(self.split_head(self.W_q(x)))
            k = self.batch_combine(self.split_head(self.W_k(x)))
            v = self.batch_combine(self.split_head(self.W_v(x)))

            # Calculate Distance/Similarity Matrix + Prime 
            rand_idx = torch.randperm(q.shape[2], device=q.device)[:self.samples]
            
            q_sample = q[:, :, rand_idx]
            
            if self.magnitude_type == 'distance':
                matrix_magnitude = self._calculate_distance_matrix_N(k, q_sample, sqrt=True)
            elif self.magnitude_type == 'similarity':
                matrix_magnitude = self._calculate_similarity_matrix_N(k, q_sample)
                
            range_idx = torch.arange(len(rand_idx), device=q.device)
                
        
            if self.magnitude_type == 'distance':
                matrix_magnitude[:, rand_idx, range_idx] = float('inf') 
            elif self.magnitude_type == 'similarity':
                matrix_magnitude[:, rand_idx, range_idx] = float('-inf')
            
            
            prime = self._prime_N(v, matrix_magnitude, self.K, rand_idx, self.maximum)
            
            # Conv1d Layer
            x = self.conv(prime)  
            
            x = self.W_o(self.combine_heads(self.batch_split(x))).permute(0, 2, 1)
            return x
    

    def _calculate_similarity_matrix(self, K, Q):
        k_norm = F.normalize(K, p=2, dim=1)
        q_norm = F.normalize(Q, p=2, dim=1)
        similarity_matrix = torch.bmm(k_norm.transpose(2, 1), q_norm)  # [B, N, M]
        similarity_matrix = torch.clamp(similarity_matrix, min=0)  # remove negative values
        return similarity_matrix
    
    def _calculate_similarity_matrix_N(self, K, Q):
        k_norm = F.normalize(K, p=2, dim=1)
        q_norm = F.normalize(Q, p=2, dim=1)
        similarity_matrix = torch.bmm(k_norm.transpose(2, 1), q_norm)  # [B, N, M]
        similarity_matrix = torch.clamp(similarity_matrix, min=0)  # remove negative values
        return similarity_matrix
        
    def _calculate_distance_matrix(self, K, Q, sqrt=False):
        norm_squared_K = torch.sum(K**2, dim=1, keepdim=True) 
        norm_squared_Q = torch.sum(Q**2, dim=1, keepdim=True) 
        
        dot_product = torch.bmm(K.transpose(2, 1), Q)  
        
        # Broadcasting: [B, 1, N] + [B, M, 1] - 2*[B, N, M]
        dist_matrix = norm_squared_K + norm_squared_Q.transpose(2, 1) - 2 * dot_product
        
        dist_matrix = torch.clamp(dist_matrix, min=0)  # remove negative values
        
        if sqrt:
            dist_matrix = torch.sqrt(dist_matrix)
        
        return dist_matrix

    def _calculate_distance_matrix_N(self, K, Q, sqrt=False):
        norm_squared_K = torch.sum(K**2, dim=1, keepdim=True).permute(0, 2, 1)
        norm_squared_Q = torch.sum(Q**2, dim=1, keepdim=True).transpose(2, 1).permute(0, 2, 1)
        
        dot_product = torch.bmm(K.transpose(2, 1), Q)  
        
        # Broadcasting: [B, 1, N] + [B, M, 1] - 2*[B, N, M]
        dist_matrix = norm_squared_K + norm_squared_Q - 2 * dot_product
        
        dist_matrix = torch.clamp(dist_matrix, min=0)  # remove negative values
        
        if sqrt:
            dist_matrix = torch.sqrt(dist_matrix)
        
        return dist_matrix

    def _prime(self, v, qk, K, maximum):
        b, c, t = v.shape 
        
        _, topk_indices = torch.topk(qk, k=K, dim=-1, largest = maximum)
        
        topk_indices_exp = topk_indices.unsqueeze(1).expand(b, c, t, K)
        
        v_expanded = v.unsqueeze(-1).expand(b, c, t, K)
        
        prime = torch.gather(v_expanded, dim=2, index=topk_indices_exp)
        
        prime = prime.reshape(b, c, -1)

        return prime
              
    def _prime_N(self, v, qk, K, rand_idx, maximum):
        b, c, t = v.shape

        # Get top-(K-1) indices from the magnitude matrix; shape: [b, t, K-1]
        _, topk_indices = torch.topk(qk, k=K - 1, dim=2, largest=maximum)
        tk = topk_indices.shape[-1]
        assert K == tk + 1, "Error: K must be same as tk + 1. K == tk + 1."

        # Map indices from the sampled space to the full token indices using rand_idx.
        # mapped_tensor will have shape: [b, t, K-1]
        mapped_tensor = rand_idx[topk_indices]

        # Create self indices for each token; shape: [1, t, 1] then expand to [b, t, 1]
        token_indices = torch.arange(t, device=v.device).view(1, t, 1).expand(b, t, 1)

        # Concatenate self index with neighbor indices to form final indices; shape: [b, t, K]
        final_indices = torch.cat([token_indices, mapped_tensor], dim=2)

        # Expand final_indices to include the channel dimension; result shape: [b, c, t, K]
        indices_expanded = final_indices.unsqueeze(1).expand(b, c, t, K)

        # Expand matrix to shape [b, c, t, 1] and then to [b, c, t, K] (ensuring contiguous memory)
        v_expanded = v.unsqueeze(-1).expand(b, c, t, K).contiguous()

        # Gather neighbor features along the token dimension (dim=2)
        prime = torch.gather(v_expanded, dim=2, index=indices_expanded)  # shape: [b, c, t, K]

        # Flatten the token and neighbor dimensions into one: [b, c, t*K]
        prime = prime.reshape(b, c, -1)
        return prime
    
class MultiHeadConv1dAttention(nn.Module):
    def __init__(self, d_model, num_heads, kernel_size): 
        super(MultiHeadConv1dAttention, self).__init__()
    
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = (self.kernel_size - 1) // 2
        
        self.W_x = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.in_channels = d_model // num_heads
        self.out_channels = d_model // num_heads
        self.conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding, 
        )
        
    def split_head(self, x): 
        batch_size, seq_length, d_model = x.size()
        self.batch_size = batch_size
        self.seq_length = seq_length
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) # (B, num_heads, seq_length, d_k)
        
    def combine_heads(self, x): 
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model) 
    
    def batch_split(self, x): 
        x = x.reshape(self.batch_size, -1, self.d_k, self.seq_length)
        return x.permute(0, 1, 3, 2).contiguous()
        
    def batch_combine(self, x): 
        batch_size, _, seq_length, d_k = x.size()
        x = x.permute(0, 1, 3, 2).contiguous() 
        return x.view(-1, self.d_k, seq_length)       
    
    def forward(self, x):
        x = self.batch_combine(self.split_head(self.W_x(x)))
        x = self.conv(x) 
        x = self.W_o(self.combine_heads(self.batch_split(x.permute(0, 2, 1))))
        return x
        
class MultiHeadConv1d(nn.Module):
    def __init__(self, d_model, num_heads, kernel_size, seq_length=197):
        super(MultiHeadConv1d, self).__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.kernel_size = kernel_size
        self.stride = 1 
        self.padding = (self.kernel_size - 1) // 2
        self.seq_length = seq_length
        
        self.W_x = nn.Linear(self.seq_length, self.seq_length)
        self.W_o = nn.Linear(self.seq_length, self.seq_length)
        
        self.in_channels = d_model // num_heads
        self.out_channels = d_model // num_heads
        self.conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding, 
        )
    
    def split_head(self, x):
        batch_size, d_model, seq_length = x.size()
        self.batch_size = batch_size
        self.seq_length = seq_length
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) # (B, num_heads, seq_length, d_k)
    
    def combine_heads(self, x): 
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, self.d_model, seq_length) 
    
    def batch_split(self, x): 
        x = x.reshape(self.batch_size, -1, self.d_k, self.seq_length)
        return x.permute(0, 1, 3, 2).contiguous()
        
    def batch_combine(self, x): 
        batch_size, _, seq_length, d_k = x.size()
        x = x.permute(0, 1, 3, 2).contiguous() 
        return x.view(-1, self.d_k, seq_length)

    def forward(self, x):    
        x = x.permute(0, 2, 1)  # Change shape to (B, seq_length, d_model)
        x = self.batch_combine(self.split_head(self.W_x(x)))
        x = self.conv(x) 
        x = self.W_o(self.combine_heads(self.batch_split(x))).permute(0, 2, 1)
        return x
    
if __name__ == "__main__":
    import torch
    from types import SimpleNamespace
    
    # ViT-Small configuration
    args = SimpleNamespace(
        img_size = (3, 224, 224),       # (channels, height, width)
        patch_size = 16,                # 16x16 patches
        num_layers = 8,                 # 8 transformer layers
        num_heads = 8,                    # 8 attention heads
        d_model = 512,                  # Hidden dimension
        num_classes = 100,              # CIFAR-100 classes
        K = 9,                          # For nearest neighbor operations
        kernel_size = 9,                # Kernel size for ConvNN
        dropout = 0.1,                  # Dropout rate
        attention_dropout = 0.1,        # Attention dropout
        num_samples = 32,                   # Sampling parameter for ConvNN
        magnitude_type = "similarity",  # Or "distance"
        shuffle_pattern = "NA",         # Default pattern
        shuffle_scale = 1,              # Default scale
        layer = "Attention",            # Attention or ConvNN
        device = torch.device("cuda" if torch.cuda.is_available() else 
                             "mps" if torch.backends.mps.is_available() else "cpu"),
        model = "ViT"                   # Model type
    )
    
    # Create the model
    model = ViT(args)
    
    print("Regular Attention")
    # Print parameter count
    total_params, trainable_params = model.parameter_count()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    x = torch.randn(64, 3, 224, 224).to(args.device)  
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}\n")
    
    
    print("ConvNN")
    args.layer = "ConvNN"
    model = ViT(args)
    total_params, trainable_params = model.parameter_count()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    output_convnn = model(x)
    
    print(f"Output shape: {output_convnn.shape}\n")
    
    
    print("ConvNNAttention")
    args.layer = "ConvNNAttention"
    model = ViT(args)
    total_params, trainable_params = model.parameter_count()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    output = model(x)
    
    print(f"Output shape: {output.shape}\n")
    

    print("Conv1d")
    args.layer = "Conv1d"
    model = ViT(args)
    total_params, trainable_params = model.parameter_count()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    output = model(x)
    
    print(f"Output shape: {output.shape}\n")
    
    
    print("Conv1dAttention")
    args.layer = "Conv1dAttention"
    model = ViT(args)
    total_params, trainable_params = model.parameter_count()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    output = model(x)
    
    print(f"Output shape: {output.shape}")