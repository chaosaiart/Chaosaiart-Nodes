import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

app.registerExtension({
	name:"chaosaiart.Show_Info",
	async beforeRegisterNodeDef(n,t,e){
		if(t.name==="chaosaiart_Show_Info"){
			function i(t){
				if(this.widgets){
					for(let e=1;e<this.widgets.length;e++)
						this.widgets[e].onRemove?.();
						this.widgets.length=1
				}
				const i=[...t];
				if(!i[0]){
					i.shift()
				}
				for(const t of i){
					const i=ComfyWidgets["STRING"](this,"text",["STRING",{
						multiline:!0
					}],e).widget;
					i.inputEl.readOnly=!0,i.inputEl.style.opacity=.6,i.value=t
				}
				requestAnimationFrame(()=>{
					const t=this.computeSize();
					t[0]<this.size[0] &&
					(t[0]=this.size[0]),t[1]<this.size[1]&&(t[1]=this.size[1]),
					this.onResize?.(t),e.graph.setDirtyCanvas(!0,!1)}
				)
			}
			const o=n.prototype.onExecuted;n.prototype.onExecuted=function(t){
				o?.apply(this,arguments),i.call(this,t.text)
			};
			const r=n.prototype.onConfigure;
			n.prototype.onConfigure=function(){
				r?.apply(this,arguments),this.widgets_values?.length&&i.call(this,this.widgets_values)
			}
		}
	}
});
