(function() {
    
    if (!globalThis.LLuL) globalThis.LLuL = {};
    
    const OBJ = (function (NAME) {
        
        let _r = 0;
        function to_gradio(v) {
            // force call `change` event on gradio
            return [v.toString(), (_r++).toString()];
        }
        
        function js2py(type, gradio_field, value) {
            // set `value` to gradio's field
            // (1) Click gradio's button.
            // (2) Gradio will fire js callback to retrieve value to be set.
            // (3) Gradio will fire another js callback to notify the process has been completed.
            return new Promise(resolve => {
                const callback_name = `${NAME}-${type}-${gradio_field}`;
                
                // (2)
                globalThis[callback_name] = () => {
                    
                    delete globalThis[callback_name];
                    
                    // (3)
                    const callback_after = callback_name + '_after';
                    globalThis[callback_after] = () => {
                        delete globalThis[callback_after];
                        resolve();
                    };
                    
                    return to_gradio(value);
                };
                
                // (1)
                gradioApp().querySelector(`#${callback_name}_set`).click();
            });
        }

        function py2js(type, pyname, ...args) {
            // call python's function
            // (1) Set args to gradio's field
            // (2) Click gradio's button
            // (3) JS callback will be kicked with return value from gradio
            
            // (1)
            return (args.length == 0 ? Promise.resolve() : js2py(type, pyname + '_args', JSON.stringify(args)))
            .then(() => {
                return new Promise(resolve => {
                    const callback_name = `${NAME}-${type}-${pyname}`;
                    // (3)
                    globalThis[callback_name] = value => {
                        delete globalThis[callback_name];
                        resolve(value);
                    }
                    // (2)
                    gradioApp().querySelector(`#${callback_name}_get`).click();
                });
            });
        }

        return { js2py, py2js }

    })('llul');

    if (!globalThis.LLuL.js2py) globalThis.LLuL.js2py = OBJ.js2py;
    if (!globalThis.LLuL.py2js) globalThis.LLuL.py2js = OBJ.py2js;

})();