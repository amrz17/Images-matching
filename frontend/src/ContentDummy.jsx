import { useState } from "react";
import TextInputDummy from "./TextInputDummy";

function ContentDummy({ licensePlate }) {
  const [matchStatus, setMatchStatus] = useState(null); // <-- pindah ke sini

  let statusContent = null;

  if (matchStatus === 'matched') {
    statusContent = (
      <div style={styles.statusContainer}>
        <img src="/correct.svg" width={160} alt="Match" />
        <h2 style={{ ...styles.statusText, color: '#63e6be' }}>MATCH</h2>
      </div>
    );
  } else if (matchStatus === 'not_matched') {
    statusContent = (
      <div style={styles.statusContainer}>
        <img src="/wrong.svg" width={160} alt="Not Match" />
        <h2 style={{ ...styles.statusText, color: '#EA0E0E' }}>
          NOT<br />MATCH
        </h2>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <div style={styles.photoSection}>
        <div style={styles.photoBox}></div>
        <div style={styles.plateBox}>
          <h2>{licensePlate}</h2>
        </div>
      </div>

      <div style={styles.scannerSection}>
        <img src="/scanner.svg" width={90} alt="Scanner" />
        <h1 style={styles.scannerText}>SCAN<br />YOUR<br />TICKET<br />ON SCANNER</h1>
        <TextInputDummy onSubmit={setMatchStatus} />
        {statusContent}
      </div>
    </div>
  );
}

// styles tetap sama
const styles = {
  container: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'flex-start',
    gap: '40px',
    marginTop: '5rem',
    width: '100%',
  },
  photoSection: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  },
  photoBox: {
    width: '360px',
    height: '400px',
    backgroundColor: '#D9D9D9',
    marginBottom: '10px',
  },
  plateBox: {
    width: '360px',
    height: '50px',
    backgroundColor: '#D9D9D9',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
  },
  scannerSection: {
    textAlign: 'center',
    marginTop: '2rem',
  },
  scannerText: {
    fontSize: '28px',
    margin: '20px 0',
  },
  statusContainer: {
    marginTop: '40px',
  },
  statusText: {
    fontSize: '40px',
    margin: '0',
  },
};

export default ContentDummy;
