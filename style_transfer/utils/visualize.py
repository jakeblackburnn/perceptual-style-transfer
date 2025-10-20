import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import math
import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os
import glob
from pathlib import Path

from style_transfer.loss import VGG
from utils.activation_extractor import ActivationExtractor

class PlotlyVisualizer:
    def __init__(self, activations_dict, image_names):
        self.activations_dict = activations_dict  # Dict mapping image names to their activations
        self.image_names = image_names  # List of available image names
        
        # Available colorscales
        self.colorscales = [
            'sunset', 'viridis', 'plasma', 'inferno', 'hot', 
            'RdBu', 'YlOrRd', 'Blues', 'Greens', 'Reds', 'Rainbow'
        ]
        
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
    def setup_layout(self):
        # Get layer options from first available image
        first_image = self.image_names[0] if self.image_names else None
        layer_options = []
        if first_image and first_image in self.activations_dict:
            layer_options = [{'label': layer, 'value': layer} for layer in self.activations_dict[first_image].keys()]
        
        # Image selection options
        image_options = [{'label': img_name, 'value': img_name} for img_name in self.image_names]
        
        # Colorscale options
        colorscale_options = [{'label': cs.title(), 'value': cs} for cs in self.colorscales]
        
        self.app.layout = html.Div([
            html.H1("Neural Network Activation Visualizer", style={'textAlign': 'center'}),
            
            # Control panel with all selection options
            html.Div([
                # First row of controls
                html.Div([
                    html.Div([
                        html.Label("Image: "),
                        dcc.Dropdown(
                            id='image-dropdown',
                            options=image_options,
                            value=self.image_names[0] if self.image_names else None,
                            style={'width': '300px'}
                        )
                    ], style={'display': 'inline-block', 'marginRight': '20px'}),
                    html.Div([
                        html.Label("Layer: "),
                        dcc.Dropdown(
                            id='layer-dropdown',
                            options=layer_options,
                            value=layer_options[0]['value'] if layer_options else None,
                            style={'width': '200px'}
                        )
                    ], style={'display': 'inline-block', 'marginRight': '20px'})
                ], style={'marginBottom': '10px'}),
                
                # Second row of controls
                html.Div([
                    html.Div([
                        html.Label("Colorscale: "),
                        dcc.Dropdown(
                            id='colorscale-dropdown',
                            options=colorscale_options,
                            value='sunset',
                            style={'width': '200px'}
                        )
                    ], style={'display': 'inline-block', 'marginRight': '20px'}),
                    html.Div([
                        html.Label("Columns: "),
                        dcc.Slider(
                            id='columns-slider',
                            min=1,
                            max=12,
                            step=1,
                            value=8,
                            marks={i: str(i) for i in range(1, 13)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], style={'display': 'inline-block', 'width': '300px', 'marginLeft': '20px'})
                ], style={'marginBottom': '10px'})
            ], style={'padding': '10px', 'textAlign': 'left'}),
            
            # Full-width scrollable graph container with dynamic height
            dcc.Graph(
                id='activation-graph', 
                style={
                    'width': '100%',
                    # Height will be calculated dynamically based on plot dimensions
                },
                config={'responsive': True}  # Enable responsive resizing
            )
        ])
    def setup_callbacks(self):
        @self.app.callback(
            Output('layer-dropdown', 'options'),
            [Input('image-dropdown', 'value')]
        )
        def update_layer_options(selected_image):
            if not selected_image or selected_image not in self.activations_dict:
                return []
            return [{'label': layer, 'value': layer} for layer in self.activations_dict[selected_image].keys()]
        
        @self.app.callback(
            Output('layer-dropdown', 'value'),
            [Input('layer-dropdown', 'options')]
        )
        def update_layer_value(layer_options):
            return layer_options[0]['value'] if layer_options else None
            
        @self.app.callback(
            Output('activation-graph', 'figure'),
            [Input('image-dropdown', 'value'), Input('layer-dropdown', 'value'), 
             Input('colorscale-dropdown', 'value'), Input('columns-slider', 'value')]
        )
        def update_visualization(selected_image, layer_name, colorscale, grid_cols):
            if not selected_image or selected_image not in self.activations_dict:
                return go.Figure()
            if not layer_name or layer_name not in self.activations_dict[selected_image]:
                return go.Figure()
            
            fig = self.create_activation_plot(selected_image, layer_name, colorscale, grid_cols)
            return fig
    
    def tensor_to_numpy(self, tensor):
        """Convert tensor to normalized numpy array for visualization"""
        tensor_np = tensor.cpu().numpy()
        tensor_np = (tensor_np - tensor_np.min()) / (tensor_np.max() - tensor_np.min() + 1e-8)
        return tensor_np
    
    def create_activation_plot(self, image_name, layer_name, colorscale, grid_cols):
        """Create Plotly subplot figure with dynamic columns and square aspect ratios"""
        activations = self.activations_dict[image_name][layer_name][0]  # Remove batch dimension
        total_channels = activations.size(0)
        
        # Calculate required rows for dynamic column layout
        grid_rows = math.ceil(total_channels / grid_cols)
        
        # Calculate spacing ratios for responsive layout
        horizontal_spacing_ratio = 0.003  # 0.3% spacing
        vertical_spacing_ratio = 0.008    # 0.8% spacing
        
        # Calculate relative height for responsive layout
        # Use base height per row that works well across different screen sizes
        base_height_per_row = 150  # pixels per row
        figure_height = grid_rows * base_height_per_row + 70  # Add margins
        
        # Use the calculated spacing ratios
        vertical_spacing = vertical_spacing_ratio
        horizontal_spacing = horizontal_spacing_ratio
        
        # Create subplot grid without titles to maximize plot area
        # We'll add channel numbers as annotations instead
        
        fig = make_subplots(
            rows=grid_rows, 
            cols=grid_cols,
            subplot_titles=None,  # No titles to maximize space
            vertical_spacing=vertical_spacing,
            horizontal_spacing=horizontal_spacing,
            shared_xaxes=False,
            shared_yaxes=False
        )
        
        # Add activation images to subplots
        for channel_idx in range(total_channels):
            row = (channel_idx // grid_cols) + 1
            col = (channel_idx % grid_cols) + 1
            
            # Convert tensor to numpy array
            activation_data = self.tensor_to_numpy(activations[channel_idx])
            
            # Create heatmap for this channel
            fig.add_trace(
                go.Heatmap(
                    z=activation_data,
                    colorscale=colorscale,
                    showscale=False,
                    hovertemplate=f"Channel {channel_idx}<br>Value: %{{z:.3f}}<extra></extra>"
                ),
                row=row, col=col
            )
        
        # Add channel number annotations positioned to avoid overlap
        for channel_idx in range(total_channels):
            row = (channel_idx // grid_cols) + 1
            col = (channel_idx % grid_cols) + 1
            
            # Calculate normalized position for annotation
            # Place in top-left corner of each subplot with small offset
            x_pos = (col - 1 + 0.05) / grid_cols
            y_pos = 1 - (row - 1 + 0.05) / grid_rows
            
            fig.add_annotation(
                text=str(channel_idx),
                x=x_pos, y=y_pos,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=10, color="white"),
                bgcolor="rgba(0,0,0,0.7)",
                bordercolor="white",
                borderwidth=1,
                xanchor="left", yanchor="top"
            )
        
        # Update layout for responsive design with calculated height
        fig.update_layout(
            title=f"{image_name} - Layer {layer_name} - All {total_channels} channels ({grid_cols} columns × {grid_rows} rows)",
            showlegend=False,
            margin=dict(l=2, r=2, t=35, b=2),  # Minimal margins
            height=figure_height,    # Set calculated height for square plots
            autosize=True           # Enable autosize for responsive width
        )
        
        # Enforce square aspect ratios and optimize space utilization
        # Use constrain='domain' to ensure axes maintain equal scaling while maximizing plot area
        for channel_idx in range(total_channels):
            row = (channel_idx // grid_cols) + 1
            col = (channel_idx % grid_cols) + 1
            
            # Get the activation data dimensions for this channel
            activation_data = self.tensor_to_numpy(activations[channel_idx])
            height, width = activation_data.shape
            
            # Calculate the subplot axis number (Plotly uses linear numbering)  
            subplot_num = (row - 1) * grid_cols + col
            
            # For first subplot, use 'y', for others use 'y2', 'y3', etc.
            y_axis_ref = 'y' if subplot_num == 1 else f'y{subplot_num}'
            
            # Configure x-axis for maximum width utilization and square aspect
            fig.update_xaxes(
                showticklabels=False, 
                showgrid=False, 
                zeroline=False,
                scaleanchor=y_axis_ref,  # Maintain square aspect ratio
                scaleratio=1,            # 1:1 aspect ratio
                constrain='domain',      # Allow full domain usage
                fixedrange=True,         # Disable zoom for consistent sizing
                range=[0, width-1],      # Set range to activation dimensions
                row=row, col=col
            )
            
            # Configure y-axis for maximum height utilization
            fig.update_yaxes(
                showticklabels=False, 
                showgrid=False, 
                zeroline=False,
                constrain='domain',      # Allow full domain usage
                fixedrange=True,         # Disable zoom for consistent sizing
                range=[0, height-1],     # Set range to activation dimensions
                autorange='reversed',    # Reverse y-axis for proper image orientation
                row=row, col=col
            )
        
        return fig
    
    def show(self):
        """Run the Dash web application"""
        self.app.run(debug=True, host='127.0.0.1', port=8050)

def visualize_activations(activations_dict, image_names):
    """Create and show the activation visualizer"""
    visualizer = PlotlyVisualizer(activations_dict, image_names)
    visualizer.show()

def generate_activations_for_image(image_path, vgg, activation_extractor, transform):
    """Generate activations for a single image"""
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Clear previous activations
        activation_extractor.activations.clear()
        
        # Run forward pass to capture activations
        with torch.no_grad():
            _ = vgg(input_tensor)
        
        return activation_extractor.activations.copy()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def load_or_generate_activations(image_files, max_images=20):
    """Load existing activations or generate new ones for multiple images"""
    activations_dir = Path('activations')
    activations_dir.mkdir(exist_ok=True)
    
    # Build perception model w/ given layers
    style_layers = ['0', '5', '10', '19', '28'] 
    content_layer = '21'
    vgg = VGG(style_layers + [content_layer]).eval() 
    
    activation_extractor = ActivationExtractor(vgg)
    
    # Load and preprocess transform
    transform = transforms.Compose([
          transforms.Resize((224, 224)),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    activations_dict = {}
    image_names = []
    
    # Limit number of images to process
    selected_images = image_files[:max_images]
    
    print(f"Processing up to {max_images} images...")
    
    for i, image_path in enumerate(selected_images):
        image_name = f"{image_path.parent.name}_{image_path.stem}"
        activation_file = activations_dir / f"activations_{image_name}.pt"
        
        print(f"({i+1}/{len(selected_images)}) Processing {image_name}...")
        
        # Try to load existing activations
        if activation_file.exists():
            try:
                activations = torch.load(activation_file)
                print(f"  Loaded existing activations from {activation_file}")
            except Exception as e:
                print(f"  Error loading {activation_file}: {e}")
                print(f"  Regenerating activations...")
                activations = generate_activations_for_image(image_path, vgg, activation_extractor, transform)
                if activations:
                    torch.save(activations, activation_file)
                    print(f"  Saved new activations to {activation_file}")
        else:
            # Generate new activations
            print(f"  Generating activations...")
            activations = generate_activations_for_image(image_path, vgg, activation_extractor, transform)
            if activations:
                torch.save(activations, activation_file)
                print(f"  Saved activations to {activation_file}")
        
        if activations:
            activations_dict[image_name] = activations
            image_names.append(image_name)
    
    # Check for legacy activations.pt file
    legacy_file = Path('activations.pt')
    if legacy_file.exists() and not activations_dict:
        print("Loading legacy activations.pt file...")
        try:
            activations = torch.load(legacy_file)
            activations_dict['legacy'] = activations
            image_names.append('legacy')
        except Exception as e:
            print(f"Error loading legacy file: {e}")
    
    return activations_dict, image_names

def main():
    # Get all content images
    image_files = [
            Path('images/content/frogs/ranitomeya-fantastica.jpeg'),
            Path('images/style/iris/01.png'),
            Path('images/Ukiyo_e/hiroshige_a-bridge-across-a-deep-gorge.jpg'),
            Path('images/Impressionism/adam-baltatu_house-on-siret-valley.jpg')
    ]
    print(f"Found {len(image_files)} content images")
    
    if not image_files:
        print("No content images found in images/content directory!")
        print("Please add some images to visualize their activations.")
        return
    
    # Load or generate activations for images
    activations_dict, image_names = load_or_generate_activations(image_files)
    
    if not activations_dict:
        print("No activations available! Please check your images and try again.")
        return
    
    print(f"\nLoaded activations for {len(image_names)} images: {image_names}")
    print(f"Available layers: {list(next(iter(activations_dict.values())).keys())}")
    
    # Launch the enhanced visualizer
    visualize_activations(activations_dict, image_names)

if __name__ == "__main__":
    main()
