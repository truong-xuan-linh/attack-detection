import torch
from transformers import AutoImageProcessor
import numpy as np
from transformers.models.vit.modeling_vit import *

class CustomViTForImageClassification(ViTPreTrainedModel):
    def __init__(self) -> None:
        
        
        vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        super().__init__(vit.config)
        self.num_labels = 2
        self.vit = vit
        # Classifier head
        self.classifier = nn.Linear(self.vit.config.hidden_size, self.num_labels)



    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class VisionTransformer:
    def __init__(self) -> None:
        self.model = torch.load("models/vit_nightshade.pt", map_location=torch.device('cpu'))
        self.model.eval()
        self.class_mapper = {
            0: "normal",
            1: "attacked"
        }
        self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    
    def predict(self, image):
        image_tensor = self.image_processor(image, return_tensors="pt")
        image_tensor = image_tensor['pixel_values']
        with torch.no_grad():
            output = self.model(pixel_values=image_tensor)
            index = np.argmax(output.logits.cpu().numpy(), axis=-1)
        return self.class_mapper[index[0]]
