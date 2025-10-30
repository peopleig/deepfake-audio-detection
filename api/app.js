const express = require('express');
// const pool = require('./db.js');
const path = require('path');
const PORT = 3000;
const app = express();

const analyze_router = require('./routes/dummyanalyze.js');

app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.urlencoded({ extended: true }));

app.get('/',(req,res) => {
    res.render('home');
})

app.use("/analyze", analyze_router);

app.listen(PORT, () => console.log(`Server active on ${PORT}`));