const express = require('express');
const multer = require('multer');
const BarcodeScanner = require('barcode-scanner');

const app = express();
const upload = multer({ dest: 'uploads/' });

app.get('/', (req, res) => {
    res.sendFile(__dirname + '/index.html');
});

app.post('/detect', upload.single('image'), (req, res) => {
    const imagePath = req.file.path;

    // Perform barcode detection using the 'barcode-scanner' library
    const scanner = new BarcodeScanner();
    const result = scanner.scan(imagePath);

    // Render the result.html page with the detected barcode image
    res.render('result.html', { barcode_image: result.barcodeImage });
});

app.listen(3000, () => {
    console.log('Server is running on port 3000');
});
