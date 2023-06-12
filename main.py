from fastapi import FastAPI
import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import os, wave, librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
import librosa
import IPython.display as ipd
import numpy as np
import pandas as pd


# Wav2vec Classification Class
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers.file_utils import ModelOutput


@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)
class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )

class Predict:
    # load model
    def load_model(self):
        model_name_or_path = "sojung1214/EmoSense"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = AutoConfig.from_pretrained(model_name_or_path)
        processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
        model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)
        self.device = device
        self.config = config
        self.processor = processor
        self.model = model
    
    # predict
    def predict(self, path, sampling_rate, processor, device, model, config):
        def speech_file_to_array_fn(path, sampling_rate):
            speech_array, _sampling_rate = torchaudio.load(path)
            resampler = torchaudio.transforms.Resample(_sampling_rate)
            speech = resampler(speech_array).squeeze().numpy()
            return speech

        speech = speech_file_to_array_fn(path, sampling_rate)
        inputs = processor.feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        inputs = {key: inputs[key].to(device) for key in inputs}
        with torch.no_grad():
            logits = model(**inputs).logits
        scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        max = 0
        for i, score in enumerate(scores):
            if max < score:
                num = i
                max = score
        outputs = {"Emotion": config.id2label[num], "Score": f"{round(max * 100, 3):.1f}%"}
        return outputs

app = FastAPI()
predictEmo = Predict()


@app.on_event("startup")
async def startup():
    predictEmo.load_model()

@app.post("/test") # 파일 업로드 후 처리하는 것으로 수정.
async def get_file(file: UploadFile):
    #파일명 만들기
    file_location = os.path.join(file.filename)
    print(file_location)
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    with open(file_location, 'rb') as opened_pcm_file:
        pcm_data = opened_pcm_file.read()
        obj2write = wave.open(file_location, 'wb')
        obj2write.setnchannels( 1 )
        obj2write.setsampwidth( 16 // 4 )
        obj2write.setframerate( 16000 )
        obj2write.writeframes( pcm_data )
        obj2write.close()
    
    # 연산
    emotion = predictEmo.predict(file_location,sampling_rate=16000, processor=predictEmo.processor, device=predictEmo.device, model=predictEmo.model, config=predictEmo.config)
      
    # 연산 후 종료
    if os.path.isfile(file_location):
        os.remove(file_location)
    return {"score": emotion['Score'], "emotion" : emotion['Emotion']}


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)