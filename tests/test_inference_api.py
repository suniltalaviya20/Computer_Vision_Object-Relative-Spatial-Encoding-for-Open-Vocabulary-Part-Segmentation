"""Exercise the browser's HTTP contract without downloading model weights."""

import base64
import io
import unittest
from unittest.mock import patch

import httpx
import torch
from PIL import Image

from deployment import inference_server as server
from inference_server import app as legacy_app


def png(mode, size, value):
    image = Image.new(mode, size, value)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class InferenceAPITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        )
        self.image = png("RGB", (8, 8), "white")
        mask = Image.new("L", (8, 8), 0)
        mask.paste(255, (2, 2, 6, 6))
        buffer = io.BytesIO()
        mask.save(buffer, format="PNG")
        self.files = {
            "image": ("image.png", self.image, "image/png"),
            "parent_mask": ("mask.png", buffer.getvalue(), "image/png"),
        }

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_entry_points_health_and_website(self):
        self.assertIs(legacy_app, server.app)
        for url in ("/", "/app.js", "/styles.css", "/data/catalogue.json",
                    "/data/inference_options.json"):
            response = await self.client.get(url)
            self.assertEqual(response.status_code, 200, url)
        health = (await self.client.get("/api/health")).json()
        self.assertEqual(health["status"], "ok")
        self.assertEqual(set(health["models"]), set(server.MODEL_SPECS))
        self.assertEqual(health["parent_categories"], list(server.PARENT_CATEGORIES))

    async def test_part_prediction_for_every_model(self):
        for model in server.MODEL_SPECS:
            with patch.object(server, "_predictor") as loader:
                loader.return_value.predict.return_value = torch.ones(8, 8)
                response = await self.client.post("/api/predict", files=self.files,
                    data={"category": " Car ", "part": " Wheel ", "model": f"final_{model}"})
                self.assertEqual(response.status_code, 200, response.text)
                body = response.json()
                self.assertEqual(body["model"], model)
                self.assertEqual(body["evaluation_mode"], "pascal_part_116")
                loader.assert_called_once_with(model)
                image, mask, query = loader.return_value.predict.call_args.args
                self.assertEqual(tuple(image.shape), (3, 8, 8))
                self.assertEqual(int(mask.sum()), 16)
                self.assertEqual(query, "wheel")
                for key in ("overlay_data_url", "mask_data_url"):
                    decoded = Image.open(io.BytesIO(base64.b64decode(body[key].split(",")[1])))
                    self.assertEqual(decoded.size, (8, 8))

    async def test_invalid_inputs_do_not_load_model(self):
        cases = [({"category": "car", "part": "wing"}, self.files, 422),
                 ({"category": "car", "part": "wheel", "model": "missing"}, self.files, 422),
                 ({"category": "car", "part": "wheel"},
                  {**self.files, "parent_mask": ("mask.png", png("L", (8, 8), 0), "image/png")}, 422),
                 ({"category": "car", "part": "wheel"},
                  {**self.files, "image": ("image.png", b"invalid", "image/png")}, 422)]
        with patch.object(server, "_predictor") as loader:
            for data, files, status in cases:
                response = await self.client.post("/api/predict", data=data, files=files)
                self.assertEqual(response.status_code, status, response.text)
            loader.assert_not_called()

    async def test_parent_detection(self):
        class Weights:
            meta = {"categories": ["__background__", "airplane"]}

        output = {"scores": torch.tensor([0.9]), "labels": torch.tensor([1]),
                  "boxes": torch.tensor([[1., 1., 6., 6.]]),
                  "masks": torch.ones(1, 1, 8, 8)}
        with patch.object(server, "_parent_predictor",
                          return_value=(lambda images: [output], Weights(), torch.device("cpu"))):
            response = await self.client.post("/api/parent/predict",
                                              files={"image": self.files["image"]})
        self.assertEqual(response.status_code, 200, response.text)
        detection = response.json()["detections"][0]
        self.assertEqual(detection["category"], "aeroplane")
        self.assertTrue(detection["mask_data_url"].startswith("data:image/png;base64,"))
