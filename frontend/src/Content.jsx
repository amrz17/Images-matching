import TextInput from "./TextInput";


function Content({ licensePlate, matchStatus}) {
    return (
      <div style={styles.container}>
        <div style={styles.photoContainer}>
            <div style={styles.photoBox}>
                <img src="" />
            </div>
            <div style={styles.plateNumber}>
                <h2>{licensePlate}</h2>
            </div>
        </div>

        
        <div style={{ marginTop: '12rem'}}>
            <img src="/scanner.svg" width={90}/>
            <h1>
                SCAN <br /> 
                YOUR <br />
                TICKET <br /> 
                ON SCANNER
            </h1>
            <TextInput />
        </div>

        {/* <div style={styles.match}>
            <img src="/correct.svg" width={160}/>
            <h2 style={{textAlign: 'center', color: '#63e6be', fontSize: '40px',
                margin: '0px'
             }}>
                Match
            </h2>
        </div>

        <div style={styles.notMatch}>
            <img src="/wrong.svg" width={160}/>
            <h2 style={{textAlign: 'center', color: '#EA0E0E', fontSize: '40px',
                margin: '0px'
             }}>
                NOT <br />
                MATCH
            </h2>

        </div> */}
        {matchStatus && (
            <div style={matchStatus === 'matched' ? styles.match : styles.notMatch}>
                <img
                src={matchStatus === 'matched' ? '/correct.svg' : '/wrong.svg'}
                width={160}
                alt={matchStatus === 'matched' ? 'Match' : 'Not Match'}
                />
                <h2
                style={{
                    textAlign: 'center',
                    color: matchStatus === 'matched' ? '#63e6be' : '#EA0E0E',
                    fontSize: '40px',
                    margin: '0px',
                }}
                >
                {matchStatus === 'matched' ? 'MATCH' : <>NOT<br />MATCH</>}
                </h2>
            </div>
            )}


        <div style={styles.photoContainer}>
            <div style={styles.photoBox}>

            </div>
            <div style={styles.plateNumber}>
                <h2>{licensePlate}</h2>

            </div>
        </div>
      </div>
    )
}

const styles = {
    container: {
        // backgroundColor: '#4CAF50',
        // backgroundColor: '#D9D9D9',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        width: '100%',
        gap: '20px'
    },

    photoContainer: {
        padding: '0px 20px',
        marginTop: '10rem',
    },

    photoBox: {
        // backgroundColor: '#888',
        backgroundColor: '#D9D9D9',
        margin: '12px',
        width: '360px',
        height: '400px'
    },
    plateNumber: {
        // backgroundColor: '#888',
        backgroundColor: '#D9D9D9',
        width: '360px',
        height: '50px',
        margin: '0 auto'
    },
    match: {
        marginTop: '140px',     
        display: 'none'
    },
    notMatch: {
        marginTop: '140px',     
        display: 'none'
    }
}

export default Content;