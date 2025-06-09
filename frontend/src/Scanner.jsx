import React, { useState } from 'react';
import QrReader from 'react-qr-reader';

function SimpleQRScanner() {
  const [scanResult, setScanResult] = useState('');
  const [scanActive, setScanActive] = useState(false);
  const [response, setResponse] = useState('');

  const handleScan = async (data) => {
    if (data) {
      setScanResult(data);
      setScanActive(false);
      await sendQRData(data);
    }
  };

  const handleError = (err) => {
    console.error(err);
    setResponse('Error saat scanning QR Code');
  };

  const sendQRData = async (qrData) => {
    try {
      const res = await fetch('http://localhost:5000/get-latest-qr', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ qr_data: qrData }),
      });

      const data = await res.json();
      if (res.ok) {
        setResponse(data.message || 'QR Code berhasil diproses');
      } else {
        setResponse(data.error || 'Terjadi kesalahan');
      }
    } catch (error) {
      console.error('Error:', error);
      setResponse('Gagal mengirim QR Code');
    }
  };

  return (
    <div className="p-4">
      <div className="flex flex-col items-center space-y-4">
        {!scanActive ? (
          <button
            onClick={() => setScanActive(true)}
            className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition"
          >
            Scan QR Code
          </button>
        ) : (
          <div className="relative w-full max-w-md">
            <button
              onClick={() => setScanActive(false)}
              className="absolute top-2 right-2 bg-red-500 text-white p-1 rounded-full z-10"
            >
              ✕
            </button>
            <QrReader
              delay={300}
              onError={handleError}
              onScan={handleScan}
              style={{ width: '100%' }}
              facingMode="environment"
            />
            <div className="mt-2 text-center text-sm text-gray-500">
              Arahkan kamera ke QR Code
            </div>
          </div>
        )}

        {scanResult && (
          <div className="mt-4 p-3 bg-gray-100 rounded">
            <p className="font-semibold">Data QR:</p>
            <p className="break-all">{scanResult}</p>
          </div>
        )}

        {response && (
          <div className={`mt-4 p-3 rounded ${response.includes('berhasil') ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
            {response}
          </div>
        )}
      </div>
    </div>
  );
}

export default SimpleQRScanner;