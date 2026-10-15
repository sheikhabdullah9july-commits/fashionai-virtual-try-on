import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import app as server


class ApplicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.paths = patch.object(server, 'UPLOAD_DIR', self.temp.name)
        self.paths.start()
        self.client = server.app.test_client()

    def tearDown(self):
        self.paths.stop()
        self.temp.cleanup()

    def test_upload_and_path_boundary(self):
        image = io.BytesIO()
        Image.new('RGB', (16, 16), 'red').save(image, 'PNG')
        image.seek(0)
        response = self.client.post('/upload', data={'file': (image, 'shirt.png')})
        self.assertEqual(response.status_code, 200)
        saved = response.json['filepath']
        self.assertEqual(server._resolve_upload(saved), str(Path(saved).resolve()))
        self.assertIsNone(server._resolve_upload(server.__file__))
        self.assertEqual(self.client.post('/api/recommend', json={'filepath': server.__file__}).status_code, 400)

    def test_invalid_upload_and_missing_inputs(self):
        response = self.client.post('/upload', data={'file': (io.BytesIO(b'not an image'), 'fake.png')})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])
        self.assertEqual(self.client.post('/upload').status_code, 400)
        self.assertEqual(self.client.post('/api/recommend', json={}).status_code, 400)
        self.assertEqual(self.client.post('/tryon', json={'uploaded_type': 'full'}).status_code, 400)
        self.assertEqual(self.client.get('/tryon/progress/missing').status_code, 404)

    def test_home_page(self):
        self.assertEqual(self.client.get('/').status_code, 200)


if __name__ == '__main__':
    unittest.main()
