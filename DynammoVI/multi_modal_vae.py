import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, scRNA_dim, ATAC_dim, SNV_dim, latent_dim, covariate_dim):
        super(Encoder, self).__init__()
        
        # Shared encoder layers for scRNA-seq and ATAC-seq data
        self.shared_encoder = nn.Sequential(
            nn.Linear(scRNA_dim + ATAC_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Shared encoder layers for SNV data
        self.SNV_shared_encoder = nn.Sequential(
            nn.Linear(SNV_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Variant encoder for incorporating covariate effects
        self.variant_encoder = nn.Sequential(
            nn.Linear(256 + 128 + covariate_dim, latent_dim),  # Combine from shared encoders and covariate
            nn.ReLU()
        )
        
        # Invariant encoder for minimizing covariate effects
        self.invariant_encoder = nn.Sequential(
            nn.Linear(256 + 128, latent_dim),  # Combine from shared encoders, excluding covariate
            nn.ReLU()
        )
        
        # Adversarial component for enforcing invariance
        self.adversarial_network = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, covariate_dim),  # Output dimension matches covariate for regression
            nn.Sigmoid()  # Assuming covariate is normalized between 0 and 1
        )

    def forward(self, scRNA_data, ATAC_data, SNV_data, covariate):
        # Process scRNA-seq and ATAC-seq data through shared encoder
        scRNA_ATAC_encoded = self.shared_encoder(torch.cat((scRNA_data, ATAC_data), dim=1))
        
        # Process SNV data through its shared encoder
        SNV_encoded = self.SNV_shared_encoder(SNV_data)
        
        # Combine encoded features
        combined_features = torch.cat((scRNA_ATAC_encoded, SNV_encoded), dim=1)
        
        # Variant latent space incorporating covariate effects
        variant_latent = self.variant_encoder(torch.cat((combined_features, covariate), dim=1))
        
        # Invariant latent space minimizing covariate effects
        invariant_latent = self.invariant_encoder(combined_features)
        
        # Adversarial prediction for enforcing invariance
        covariate_pred = self.adversarial_network(invariant_latent)

        return variant_latent, invariant_latent, covariate_pred


class Decoder(nn.Module):
    def __init__(self, latent_dim, scRNA_dim, ATAC_dim, SNV_dim):
        super(Decoder, self).__init__()
        
        # Decoder for scRNA-seq data from the variant latent space
        self.scRNA_decoder_variant = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, scRNA_dim),
            nn.Softplus()  # Ensure non-negative output for count data
        )
        
        # Decoder for ATAC-seq data from the variant latent space
        self.ATAC_decoder_variant = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, ATAC_dim),
            nn.Softplus()  # Ensure non-negative output for count data
        )
        
        # Decoder for SNV data from the variant latent space
        self.SNV_decoder_variant = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, SNV_dim),
            nn.Sigmoid()  # Output probabilities for binary data
        )
        
        # Decoders for invariant latent space can be similarly defined if needed

    def forward(self, latent_variant):
        # Decode scRNA-seq data from variant latent space
        scRNA_recon_variant = self.scRNA_decoder_variant(latent_variant)
        
        # Decode ATAC-seq data from variant latent space
        ATAC_recon_variant = self.ATAC_decoder_variant(latent_variant)
        
        # Decode SNV data from variant latent space
        SNV_recon_variant = self.SNV_decoder_variant(latent_variant)
        
        # Decoding from invariant latent space can be added if required
        
        return scRNA_recon_variant, ATAC_recon_variant, SNV_recon_variant

