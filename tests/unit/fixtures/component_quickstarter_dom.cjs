// Execute unchanged page scripts against the DOM operations they use.
// This fixture checks behavior and does not claim a rendered browser result.
const fs = require('node:fs');
const vm = require('node:vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));

class Element {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.value = '';
    this.name = '';
    this.id = '';
    this.classes = new Set();
    this.listeners = {};
    this.innerHTML = '';
    this._text = '';
    this.classList = {
      add: (...names) => names.forEach(name => this.classes.add(name))
    };
  }

  set textContent(value) {
    this._text = value;
    this.innerHTML = '';
  }

  get textContent() { return this._text; }
  get lastChild() { return this.children.at(-1); }

  appendChild(child) {
    this.children.push(child);
    return child;
  }

  insertBefore(child, existing) {
    this.children.splice(this.children.indexOf(existing), 0, child);
  }

  add(option) {
    this.children.push(option);
    if (this.children.length === 1) this.value = option.value;
  }

  addEventListener(name, listener) {
    (this.listeners[name] ??= []).push(listener);
  }

  querySelectorAll(selector) {
    const matches = element => {
      if (selector === 'input') return element.tag === 'input';
      if (selector === '.register') return element.classes.has('register');
      const attribute = selector.match(/name="([^"]+)"/);
      return attribute ? element.name === attribute[1] : false;
    };
    return this.children.flatMap(child => [
      ...(matches(child) ? [child] : []),
      ...child.querySelectorAll(selector)
    ]);
  }

  querySelector(selector) { return this.querySelectorAll(selector)[0]; }
}

const root = new Element('document');
const byId = {};
const html = fs.readFileSync(input.path, 'utf8');
for (const match of html.matchAll(/<([a-z]+)\b([^>]*\bid="[^"]+"[^>]*)>/g)) {
  const attributes = match[2];
  const node = new Element(match[1]);
  node.id = attributes.match(/\bid="([^"]+)"/)[1];
  node.value = attributes.match(/\bvalue="([^"]*)"/)?.[1] ?? '';
  byId[node.id] = node;
  root.appendChild(node);
}

const context = {
  document: {
    createElement: tag => new Element(tag),
    getElementById: id => byId[id],
    querySelectorAll: selector => root.querySelectorAll(selector)
  },
  Option: function(value) {
    const node = new Element('option');
    node.value = value;
    return node;
  }
};
if (input.highlight !== false) {
  context.hljs = {highlight: code => ({value: code})};
}
vm.createContext(context);
for (const match of html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)) {
  if (!/\bsrc=/.test(match[1])) vm.runInContext(match[2], context);
}

const read = () => byId.template_code.innerHTML || byId.template_code.textContent;
const initial = read();
if (input.registers) {
  byId.registers.children = [];
  for (const register of input.registers) {
    context.addRegisterBoxes(register.reg_name, register.bitsize, register.desc);
    byId.registers.lastChild.querySelector('[name="side"]').value = register.side;
  }
}
if (input.className !== undefined) {
  (byId.component_name ?? byId.bloq_name).value = input.className;
}
if (input.description !== undefined) {
  byId.one_line_desc.value = input.description;
}
context.generateCode();
for (const action of input.actions ?? []) {
  const node = action.register === undefined
    ? byId[action.id]
    : byId.registers.children[action.register].querySelector(`[name="${action.name}"]`);
  if (action.value !== undefined) node.value = action.value;
  for (const listener of node.listeners[action.event] ?? []) listener();
}

process.stdout.write(JSON.stringify({
  initial,
  code: read(),
  registers: context.getRegisterInputs(),
  events: root.querySelectorAll('input').map(node => Object.keys(node.listeners)),
  error: byId.error?.textContent ?? ''
}));
