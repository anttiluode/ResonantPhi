import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List
import numpy as np
import gc
import traceback
from tqdm import tqdm

class WaveNeuronLayer(nn.Module):
    def __init__(self, hidden_size: int, wave_units: int = 16, coupling_density: float = 0.75):
        super().__init__()
        self.hidden_size = hidden_size
        self.wave_units = wave_units
        self.coupling_density = coupling_density  # Control density of connections

        # Wave parameters
        self.frequencies = nn.Parameter(torch.rand(wave_units) * 2.0)
        self.phases = nn.Parameter(torch.rand(wave_units) * 2 * np.pi)

        # Memory state
        self.register_buffer('time_step', torch.tensor(0.))
        self.register_buffer('wave_state', torch.zeros(wave_units))
        self.register_buffer('resonance', torch.zeros(wave_units))

        # Neural projections
        self.input_proj = nn.Sequential(
            nn.Linear(hidden_size, wave_units * 2),
            nn.LayerNorm(wave_units * 2),
            nn.SiLU(),
            nn.Linear(wave_units * 2, wave_units)
        )

        self.output_proj = nn.Sequential(
            nn.Linear(wave_units, wave_units * 2),
            nn.LayerNorm(wave_units * 2),
            nn.SiLU(),
            nn.Linear(wave_units * 2, hidden_size)
        )

        # Dendrite (coupling) matrix with increased density
        coupling_matrix = torch.randn(wave_units, wave_units) * 0.1
        mask = torch.rand(wave_units, wave_units) < self.coupling_density  # Control density
        self.coupling = nn.Parameter(coupling_matrix * mask)

        # Damping factors
        self.damping = nn.Parameter(torch.ones(wave_units) * 0.1)

    def compute_wave_dynamics(self, x: torch.Tensor) -> torch.Tensor:
        """Compute wave dynamics with coupling."""
        batch_size, seq_len, _ = x.shape

        # Update time
        self.time_step += 0.1
        t = self.time_step

        # Project input to wave space
        wave_input = self.input_proj(x.float())  # [batch, seq, wave_units]

        # Generate base waves with damping
        base_waves = torch.sin(2 * np.pi * self.frequencies * t + self.phases) * \
                     torch.exp(-self.damping * t)

        # Expand base waves to match input dimensions
        base_waves = base_waves.view(1, 1, -1).expand(batch_size, seq_len, -1)

        # Apply coupling between waves with increased dendritic connections
        coupled_waves = torch.tanh(
            torch.matmul(base_waves, self.coupling.unsqueeze(0))
        )

        # Modulate with input
        modulated_waves = coupled_waves * (1 + wave_input)

        # Update resonance state with mean across batch and sequence
        mean_waves = modulated_waves.mean(dim=[0, 1])  # Average over batch and sequence
        self.resonance = 0.9 * self.resonance + 0.1 * torch.abs(mean_waves)

        # Update wave state
        self.wave_state = 0.9 * self.wave_state + 0.1 * mean_waves

        # Project back to hidden space
        output = self.output_proj(modulated_waves)

        return output.to(x.dtype)

    def reset_state(self):
        """Reset all dynamic states."""
        self.time_step.zero_()
        self.wave_state.zero_()
        self.resonance.zero_()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Process hidden states through wave dynamics."""
        wave_output = self.compute_wave_dynamics(hidden_states)
        
        # Apply resonance-modulated residual connection
        resonance_factor = torch.sigmoid(self.resonance.mean())
        output = hidden_states + wave_output * resonance_factor

        return output

class WaveEnhancedPhi(nn.Module):
    def __init__(
        self,
        base_model_name: str = "microsoft/phi-2",
        wave_units: int = 16,
        device: str = "cuda"
    ):
        super().__init__()
        self.device = device

        print("Loading base model...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Set pad_token to eos_token")

        # Load model with quantization
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=["lm_head", "wave_layer"]
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )

        # Add wave neuron layer
        self.wave_layer = WaveNeuronLayer(
            hidden_size=self.model.config.hidden_size,
            wave_units=wave_units,
            coupling_density=0.75  # Set density for more dendrites
        ).to(device)

        print(f"Model enhanced with {wave_units} wave neurons")

    def clear_memory(self):
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_length: int = 150,
        temperature: float = 0.8,
        top_p: float = 0.92,
        show_progress: bool = True
    ) -> List[str]:
        try:
            print(f"\nGenerating response for: {prompt[:50]}...")

            # Reset wave states
            self.wave_layer.reset_state()

            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                return_attention_mask=True,
                padding=True,
                truncation=True
            ).to(self.device)

            # Set up progress bar
            if show_progress:
                pbar = tqdm(total=max_length - inputs.input_ids.shape[1])

            # Generate tokens
            for _ in range(max_length - inputs.input_ids.shape[1]):
                # Get model outputs
                outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True
                )

                # Process through wave layer
                hidden_states = outputs.hidden_states[-1]
                wave_hidden = self.wave_layer(hidden_states)

                # Get logits and sample next token
                logits = self.model.lm_head(wave_hidden)
                next_token_logits = logits[:, -1, :] / temperature

                # Apply top-p sampling
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits.masked_fill_(indices_to_remove, float('-inf'))

                # Sample token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Update sequence
                inputs.input_ids = torch.cat([inputs.input_ids, next_token], dim=1)
                inputs.attention_mask = torch.cat([
                    inputs.attention_mask,
                    torch.ones((1, 1), dtype=torch.long, device=self.device)
                ], dim=1)

                if show_progress:
                    pbar.update(1)

                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

            if show_progress:
                pbar.close()

            # Decode output
            generated_texts = self.tokenizer.batch_decode(
                inputs.input_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            return [text.strip() for text in generated_texts]

        except Exception as e:
            print(f"Generation error: {str(e)}")
            traceback.print_exc()
            return [prompt]
        finally:
            self.clear_memory()

def test_wave_phi():
    """Test the wave-enhanced Phi model."""
    print("Initializing Wave-Enhanced Phi...")
    model = WaveEnhancedPhi(
        wave_units=16  # Adjust wave units as desired
    )

    test_prompts = [
        "Write a surreal story about a robot discovering emotions:",
        "Describe a color that doesn't exist:",
        "Create a recipe for happiness:",
        "Write a conversation between two quantum particles:",
    ]

    print("\nGenerating responses...")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        try:
            responses = model.generate(
                prompt,
                max_length=200,
                temperature=0.85,
                top_p=0.92
            )
            print("\nResponse:", responses[0])
        except Exception as e:
            print(f"Error generating response: {str(e)}")
        finally:
            print("\nClearing memory...")
            model.clear_memory()

    print("\nTest complete!")

if __name__ == "__main__":
    test_wave_phi()
