function Content() {
    return (
      <div style={styles.container}>
        <div style={styles.photoContainer}>
            <div style={styles.photoBox}>

            </div>
            <div style={styles.plateNumber}>

            </div>
        </div>

        <div>
            <h2>
                SCAN <br /> 
                Your <br />
                TICKET <br /> 
                ON SCANNER
            </h2>
            <img src="../public/scanner.svg" width={90}/>
        </div>

        <div style={styles.photoContainer}>
            <div style={styles.photoBox}>

            </div>
            <div style={styles.plateNumber}>

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
        gap: '24px'
    },

    photoContainer: {
        padding: '0px 20px',
        marginTop: '10rem',
    },

    photoBox: {
        backgroundColor: '#888',
        margin: '12px',
        width: '360px',
        height: '400px'
    },
    plateNumber: {
        backgroundColor: '#888',
        width: '360px',
        height: '50px',
        margin: '0 auto'
    },

}

export default Content;