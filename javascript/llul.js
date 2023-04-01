(function () {

    if (!globalThis.LLuL) globalThis.LLuL = {};
    const LLuL = globalThis.LLuL;

    function id(type, s) {
        return `llul-${type}-${s}`;
    }

    function isDark() {
        return gradioApp().querySelector('.dark') !== null;
    }

    const M = 2;
    function setSize(canvas, width, height) {
        width = Math.floor(+width / M);
        height = Math.floor(+height / M);
        if (canvas.width != width) canvas.width = width;
        if (canvas.height != height) canvas.height = height;
    }

    function updateXY(canvas) {
        let x = +canvas.dataset.x,
            y = +canvas.dataset.y,
            m = +canvas.dataset.m,
            mm = Math.pow(2, m),
            w = +canvas.width,
            h = +canvas.height;
        if (x < 0) x = 0;
        if (w < x + w / mm) x = Math.floor(w - w / mm);
        if (y < 0) y = 0;
        if (h < y + h / mm) y = Math.floor(h - h / mm);

        canvas.dataset.x = x;
        canvas.dataset.y = y;
        canvas.dataset.m = m;
    }

    function draw(canvas) {
        const
            x = +canvas.dataset.x,
            y = +canvas.dataset.y,
            m = +canvas.dataset.m,
            mm = Math.pow(2, m),
            w = +canvas.width,
            h = +canvas.height;

        const ctx = canvas.getContext('2d');

        const bgcolor = isDark() ? 'black' : 'white';
        ctx.fillStyle = bgcolor;
        ctx.fillRect(0, 0, +canvas.width, +canvas.height);

        ctx.fillStyle = 'gray';
        ctx.fillRect(x, y, Math.floor(w / mm), Math.floor(h / mm));
    }

    async function update_gradio(type, canvas) {
        await LLuL.js2py(type, 'x', +canvas.dataset.x * M);
        await LLuL.js2py(type, 'y', +canvas.dataset.y * M);
    }

    function init(type) {
        const $ = x => gradioApp().querySelector(x);
        const cont = $('#' + id(type, 'container'));
        const x = $('#' + id(type, 'x'));
        const y = $('#' + id(type, 'y'));
        const m = $(`#${id(type, 'm')} input[type=number]`);
        const ms = $(`#${id(type, 'm')} input[type=range]`);
        if (!cont || !x || !y || !m || !ms) return false;

        const width = $(`#${type}_width input[type=number]`);
        const height = $(`#${type}_height input[type=number]`);
        const width2 = $(`#${type}_width input[type=range]`);
        const height2 = $(`#${type}_height input[type=range]`);

        const canvas = document.createElement('canvas');
        canvas.style.border = '1px solid gray';
        canvas.dataset.x = Math.floor(+width.value / 4 / M);
        canvas.dataset.y = Math.floor(+height.value / 4 / M);
        canvas.dataset.m = m.value;

        for (let ele of [width, height, width2, height2, m, ms]) {
            ele.addEventListener('input', e => {
                canvas.dataset.m = +m.value;
                setSize(canvas, width.value, height.value);
                updateXY(canvas);
                draw(canvas);
            });
        }

        let dragging = false;
        let last_x, last_y;
        canvas.addEventListener('pointerdown', e => {
            e.preventDefault();
            dragging = true;
            last_x = e.offsetX;
            last_y = e.offsetY;
        });
        canvas.addEventListener('pointerup', async e => {
            e.preventDefault();
            dragging = false;
            await update_gradio(type, canvas);
        });
        canvas.addEventListener('pointermove', e => {
            if (!dragging) return;
            const dx = e.offsetX - last_x, dy = e.offsetY - last_y;
            const x = +canvas.dataset.x, y = +canvas.dataset.y;
            canvas.dataset.x = x + dx;
            canvas.dataset.y = y + dy;
            last_x = e.offsetX;
            last_y = e.offsetY;
            updateXY(canvas);
            draw(canvas);
        });

        cont.appendChild(canvas);
        setSize(canvas, width.value, height.value);
        updateXY(canvas);
        draw(canvas);

        return true;
    }

    function init_LLuL() {
        if (!LLuL.txt2img) {
            LLuL.txt2img = init('txt2img');
            if (LLuL.txt2img) {
                console.log('[LLuL] txt2img initialized');
            }
        }

        if (!LLuL.img2img) {
            LLuL.img2img = init('img2img');
            if (LLuL.img2img) {
                console.log('[LLuL] img2img initialized');
            }
        }

        return LLuL.txt2img && LLuL.img2img;
    }

    function apply() {
        const ok = init_LLuL();
        if (!ok) {
            setTimeout(apply, 500);
        }
    }

    apply();

})();
