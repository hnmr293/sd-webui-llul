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

    let last_image = new Image();
    async function draw(canvas) {
        const
            x = +canvas.dataset.x,
            y = +canvas.dataset.y,
            m = +canvas.dataset.m,
            mm = Math.pow(2, m),
            w = +canvas.width,
            h = +canvas.height,
            bg = canvas.dataset.bg;

        const ctx = canvas.getContext('2d');

        if (bg) {
            if (last_image?.src === bg) {
                // do nothing
            } else {
                await (new Promise(resolve => {
                    last_image.onload = () => resolve();
                    last_image.src = bg;
                }));
            }
        }
        
        if (last_image.src) {
            ctx.drawImage(last_image, 0, 0, +last_image.width, +last_image.height, 0, 0, +canvas.width, +canvas.height);
        } else {
            const bgcolor = isDark() ? 'black' : 'white';
            ctx.fillStyle = bgcolor;
            ctx.fillRect(0, 0, +canvas.width, +canvas.height);
        }

        ctx.fillStyle = 'gray';
        ctx.fillRect(x, y, Math.floor(w / mm), Math.floor(h / mm));
    }

    async function update_gradio(type, canvas) {
        await LLuL.js2py(type, 'x', +canvas.dataset.x * M);
        await LLuL.js2py(type, 'y', +canvas.dataset.y * M);
    }

    function init(type) {
        const $ = x => Array.from(gradioApp().querySelectorAll(x)).at(-1);
        const cont = $('#' + id(type, 'container'));
        const x = $('#' + id(type, 'x'));
        const y = $('#' + id(type, 'y'));
        const m = $(`#${id(type, 'm')} input[type=number]`);
        const ms = $(`#${id(type, 'm')} input[type=range]`);
        if (!cont || !x || !y || !m || !ms) return false;

        if (cont.querySelector('canvas')) return true; // already called

        const width = $(`#${type}_width input[type=number]`);
        const height = $(`#${type}_height input[type=number]`);
        const width2 = $(`#${type}_width input[type=range]`);
        const height2 = $(`#${type}_height input[type=range]`);

        const canvas = document.createElement('canvas');
        canvas.style.border = '1px solid gray';
        canvas.dataset.x = Math.floor(+width.value / 4 / M);
        canvas.dataset.y = Math.floor(+height.value / 4 / M);
        canvas.dataset.m = m.value;

        const bg_cont = document.createElement('div');
        bg_cont.classList.add('llul-bg-setting');
        bg_cont.innerHTML = `
<span>Load BG</span>
<span>Erase BG</span>
<input type="file" style="display:none">
        `;

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
        
        function set_bg(url) {
            canvas.dataset.bg = url;
            draw(canvas);
        }
        bg_cont.querySelector('input[type=file]').addEventListener('change', e => {
            const ele = e.target;
            const files = ele.files;
            if (files.length != 0) {
                const file = files[0];
                const r = new FileReader();
                r.onload = () => set_bg(r.result);
                r.readAsDataURL(file);
            }
            ele.value = '';
        }, false);
        bg_cont.addEventListener('click', e => {
            const ele = e.target;
            if (ele.textContent == 'Load BG') {
                bg_cont.querySelector('input[type=file]').click();
            } else if (ele.textContent == 'Erase BG') {
                set_bg('');
            }
        });

        cont.appendChild(canvas);
        cont.appendChild(bg_cont);
        setSize(canvas, width.value, height.value);
        updateXY(canvas);
        draw(canvas);

        return true;
    }

    function init2(type, init_fn) {
        const get_acc = new Promise(resolve => {
            (function try_get_acc() {
                const acc = gradioApp().querySelector('#' + id(type, 'accordion'));
                if (acc) {
                    resolve(acc);
                } else {
                    setTimeout(try_get_acc, 500);
                }
            })();
        });

        return get_acc.then(acc => {
            const observer = new MutationObserver((list, observer) => {
                for (let mut of list) {
                    //console.log(mut.type);
                    if (mut.type === 'childList') {
                        //console.log(mut.addedNodes);
                        //console.log(mut.removedNodes);
                        if (mut.addedNodes.length != 0) {
                            // closed -> opened
                            init_fn(type);
                        } else {
                            // opened -> closed
                            // do nothing
                        }
                    }
                }
            });
            observer.observe(acc, { childList: true, attributes: false, subtree: false });
        });
    }
    
    function init_LLuL() {
        if (!LLuL.txt2img) {
            LLuL.txt2img = init2('txt2img', init);
            if (LLuL.txt2img) {
                LLuL.txt2img.then(() => console.log('[LLuL] txt2img initialized'));
            }
        }

        if (!LLuL.img2img) {
            LLuL.img2img = init2('img2img', init);
            if (LLuL.img2img) {
                LLuL.img2img.then(() => console.log('[LLuL] img2img initialized'));
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
